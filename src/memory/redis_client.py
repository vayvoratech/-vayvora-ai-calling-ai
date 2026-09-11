import os

from dotenv import load_dotenv
from redis import Redis


# Load environment variables from the project root .env
load_dotenv()


REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379",
)


redis_client = Redis.from_url(
    REDIS_URL,
    decode_responses=True,
)