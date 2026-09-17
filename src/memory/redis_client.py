import os
import redis

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Standalone Redis client with connection pooling
redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=False,
    socket_timeout=5,
)
