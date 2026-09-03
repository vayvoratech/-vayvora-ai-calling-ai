from redis_client import redis_client
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery

rag_schema_dict = {
    "index": {"name": "rag_chunks_idx", "prefix": "chunk:", "storage_type": "hash"},
    "fields": [
        {"name": "text", "type": "text"},
        {"name": "vector", "type": "vector", "attrs": {"algorithm": "HNSW", "datatype": "FLOAT32", "dim": 1536, "distance_metric": "COSINE"}}
    ]
}

rag_index = SearchIndex(IndexSchema.from_dict(rag_schema_dict), redis_client)

def init_vector_db():
    rag_index.create(overwrite=False)

def search_vector_db(query_vector: list, top_k: int = 3):
    query = VectorQuery(vector=query_vector, vector_field_name="vector", num_results=top_k, return_fields=["text"])
    return rag_index.query(query)