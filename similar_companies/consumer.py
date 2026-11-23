"""Consume company events from RabbitMQ and keep Postgres in sync with the registry.

Per message: validate -> skip if not newer -> embed -> newer-wins write -> refresh the
neighbour cache -> commit -> ack.

The database commit always happens before the ack. A crash in between means RabbitMQ
redelivers the message, never that a company is lost, and a redelivery is harmless
because every write is newer-wins (an equal timestamp is a no-op).

Failures are split three ways:
- bad data: stored in rejected_events with its reason, then acked;
- an unexpected error on a valid message (e.g. the embedding model failing): the model
  call is retried first; if it still fails, the message is quarantined as
  `processing_error` and acked, so one poison message can never block the queue. Those
  messages are replayed with `uv run python -m scripts.reprocess_rejected`;
- database unreachable: the message goes back on the queue and the consumer waits for
  the database to return before taking more.
"""

import asyncio
import hashlib
import logging
import time
from collections import Counter

import aio_pika
import psycopg

from similar_companies import neighbours, repository
from similar_companies.config import Settings, load_settings
from similar_companies.embedding import Embedder, create_embedder, embedding_text
from similar_companies.validation import Rejection, StruckOffEvent, UpsertEvent, parse_event

logger = logging.getLogger(__name__)

QUEUE = "company-events"  # owned by the registry team: declared exactly as the producer does
APPLIED, SKIPPED, REJECTED = "applied", "skipped", "rejected"
PROCESSING_ERROR = "processing_error"
TRANSIENT_ERRORS = (psycopg.OperationalError, psycopg.InterfaceError)
MAX_RECONNECT_WAIT_SECONDS = 30
PROGRESS_EVERY = 500
EMBED_RETRY_DELAYS = (0.5, 2.0)  # seconds to wait before the 2nd and 3rd embedding attempt


class MessageHandler:
    """Processes one message body in one database transaction. Knows nothing about RabbitMQ."""

    def __init__(
        self,
        database_url: str,
        embedder: Embedder,
        embed_retry_delays: tuple[float, ...] = EMBED_RETRY_DELAYS,
    ) -> None:
        self.database_url = database_url
        self.embedder = embedder
        self.embed_retry_delays = embed_retry_delays
        self.conn = psycopg.connect(database_url, autocommit=True)

    def handle(self, body: bytes, message_id: str) -> str:
        """Return APPLIED, SKIPPED or REJECTED. Raises TRANSIENT_ERRORS if the database is down."""
        try:
            with self.conn.transaction():
                return self._process(body, message_id)
        except TRANSIENT_ERRORS:
            raise
        except Exception:
            # A bug, or a model failure that outlasted the retries. Park it with the raw body
            # instead of crashing or redelivering it forever; replay it after the fix with
            # scripts.reprocess_rejected.
            logger.exception(
                "unexpected error on message %s; quarantined as %s", message_id, PROCESSING_ERROR
            )
            with self.conn.transaction():
                repository.save_rejected(self.conn, message_id, decode(body), PROCESSING_ERROR)
            return REJECTED

    def reprocess_rejected(self, reason: str = PROCESSING_ERROR) -> Counter:
        """Replay quarantined messages with this reason through the normal handling.

        Safe to run at any time: writes are newer-wins, so a message overtaken by a newer
        update is a no-op. Each message gets one transaction: its quarantine row is deleted
        only if processing succeeds; if it fails again, everything rolls back and it stays.
        """
        outcomes: Counter = Counter()
        for rejected_id, message_id, raw_body in repository.rejected_messages(self.conn, reason):
            try:
                with self.conn.transaction():
                    repository.delete_rejected(self.conn, rejected_id)
                    outcome = self._process(raw_body.encode("utf-8"), message_id)
                    if outcome == REJECTED:
                        raise _StillRejected
            except _StillRejected:
                outcome = REJECTED  # still invalid; its reason was logged by _process
            except TRANSIENT_ERRORS:
                raise
            except Exception:
                logger.exception("message %s still fails; left in quarantine", message_id)
                outcome = REJECTED
            outcomes[outcome] += 1
        return outcomes

    def reconnect(self) -> None:
        """Replace the database connection, waiting with exponential backoff until it works."""
        wait = 1
        while True:
            try:
                self.conn.close()
                self.conn = psycopg.connect(self.database_url, autocommit=True)
                logger.info("database connection restored")
                return
            except psycopg.OperationalError:
                logger.warning("database still unavailable; retrying in %ss", wait)
                time.sleep(wait)
                wait = min(wait * 2, MAX_RECONNECT_WAIT_SECONDS)

    def _process(self, body: bytes, message_id: str) -> str:
        event = parse_event(body)
        if isinstance(event, Rejection):
            logger.warning("rejected %s: %s (%s)", message_id, event.reason, event.detail)
            repository.save_rejected(self.conn, message_id, decode(body), event.reason)
            return REJECTED

        stored = repository.stored_state(self.conn, event.registration_number)
        if stored and event.updated_at <= stored.updated_at:
            return SKIPPED  # duplicate or stale: nothing newer to store, so no embedding work

        if isinstance(event, StruckOffEvent):
            applied = repository.mark_struck_off(
                self.conn, event.registration_number, event.updated_at
            )
        else:
            if stored and stored.status == "struck_off":
                logger.warning("reactivating struck-off company %s", event.registration_number)
            applied = repository.upsert_company(self.conn, self._to_row(event))
        if not applied:
            return SKIPPED
        neighbours.refresh_after_change(self.conn, event.registration_number)  # same transaction
        return APPLIED

    def _embed(self, text: str) -> list[float]:
        """Embed one text, retrying briefly so a temporary model failure doesn't quarantine
        a valid message. The last failure propagates."""
        for delay in self.embed_retry_delays:
            try:
                return self.embedder.embed([text])[0]
            except Exception:
                logger.warning("embedding failed; retrying in %ss", delay, exc_info=True)
                time.sleep(delay)
        return self.embedder.embed([text])[0]

    def _to_row(self, event: UpsertEvent) -> repository.CompanyUpsert:
        text = embedding_text(event)
        vector = self._embed(text)
        return repository.CompanyUpsert(
            registration_number=event.registration_number,
            updated_at=event.updated_at,
            company_name=event.company_name,
            nature_of_business=event.nature_of_business,
            state=event.state,
            industry=event.industry,
            embedded_text=text,
            embedding=vector,
            embedding_model=self.embedder.model_name,
        )


class _StillRejected(Exception):
    """A reprocessed message failed again; used to roll back its savepoint."""


def decode(body: bytes) -> str:
    """Message body as text for the quarantine table, even if it isn't valid UTF-8."""
    return body.decode("utf-8", errors="replace")


def message_id_of(message: aio_pika.abc.AbstractIncomingMessage) -> str:
    """The producer's message_id, or a hash of the body if a message arrives without one."""
    return message.message_id or hashlib.sha256(message.body).hexdigest()


async def process(
    message: aio_pika.abc.AbstractIncomingMessage, handler: MessageHandler, counts: Counter
) -> None:
    message_id = message_id_of(message)
    try:
        # The handler is synchronous (psycopg, model inference); a worker thread keeps the
        # event loop free, so RabbitMQ heartbeats continue during slow steps.
        outcome = await asyncio.to_thread(handler.handle, message.body, message_id)
    except TRANSIENT_ERRORS:
        logger.exception("database unavailable; returning %s to the queue", message_id)
        await message.nack(requeue=True)
        await asyncio.to_thread(handler.reconnect)
        return
    await message.ack()  # only after the transaction has committed
    counts[outcome] += 1
    if counts.total() % PROGRESS_EVERY == 0:
        logger.info("processed %s messages: %s", counts.total(), dict(counts))


async def consume(settings: Settings, handler: MessageHandler) -> None:
    connection = await aio_pika.connect_robust(settings.rabbitmq_url)  # reconnects by itself
    async with connection:
        channel = await connection.channel()
        await channel.set_qos(prefetch_count=settings.prefetch_count)
        queue = await channel.declare_queue(QUEUE, durable=True)
        logger.info("consuming from %s", QUEUE)
        counts: Counter = Counter()
        async with queue.iterator() as messages:
            async for message in messages:
                await process(message, handler, counts)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = load_settings()
    logger.info("loading embedding model %s", settings.embedding_model)
    embedder = create_embedder(settings.embedding_model)  # once, before taking any message
    handler = MessageHandler(settings.database_url, embedder)
    repository.apply_schema(handler.conn)
    if not neighbours.is_complete(handler.conn):
        logger.info("building the neighbour cache for existing companies")
        neighbours.rebuild_all(handler.conn)
    try:
        asyncio.run(consume(settings, handler))
    except KeyboardInterrupt:
        logger.info("stopped; unacked messages return to the queue")


if __name__ == "__main__":
    main()
