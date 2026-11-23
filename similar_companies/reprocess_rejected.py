"""Replay quarantined messages after a fix, e.g. once a failing embedding model works again.

Each message goes through the consumer's normal handling. Writes are newer-wins, so a
message that has since been overtaken by a newer update changes nothing. A message that
succeeds leaves the quarantine; one that fails again stays there.
"""

import argparse
import logging

from similar_companies.config import load_settings
from similar_companies.consumer import PROCESSING_ERROR, MessageHandler
from similar_companies.embedding import create_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reason",
        default=PROCESSING_ERROR,
        help="which quarantined messages to replay (a reason code from rejected_events)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    handler = MessageHandler(settings.database_url, create_embedder(settings.embedding_model))
    outcomes = handler.reprocess_rejected(args.reason)
    print(f"reprocessed {outcomes.total()} '{args.reason}' messages: {dict(outcomes)}")


if __name__ == "__main__":
    main()
