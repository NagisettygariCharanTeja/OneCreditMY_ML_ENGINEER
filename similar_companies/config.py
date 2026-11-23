import os
from dataclasses import dataclass

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@dataclass(frozen=True)
class Settings:
    database_url: str
    rabbitmq_url: str
    embedding_model: str
    prefetch_count: int  # messages RabbitMQ may hand the consumer before they are acked


def load_settings() -> Settings:
    postgres_port = os.environ.get("POSTGRES_PORT", "5432")
    rabbitmq_port = os.environ.get("RABBITMQ_PORT", "5672")
    return Settings(
        database_url=os.environ.get(
            "DATABASE_URL", f"postgresql://postgres:postgres@localhost:{postgres_port}/companies"
        ),
        rabbitmq_url=os.environ.get(
            "RABBITMQ_URL", f"amqp://guest:guest@localhost:{rabbitmq_port}/"
        ),
        embedding_model=os.environ.get("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        prefetch_count=int(os.environ.get("PREFETCH_COUNT", "20")),
    )
