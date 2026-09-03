from redis_client import redis_client
from redisvl.extensions.cache.llm import SemanticCache

ai_semantic_cache = SemanticCache(
    name="ai_call_cache",
    redis_client=redis_client,
    distance_threshold=0.15,
    ttl=3600
)

def check_cache(prompt: str):
    return ai_semantic_cache.check(prompt=prompt)

def store_in_cache(prompt: str, response: str):
    ai_semantic_cache.store(prompt=prompt, response=response)