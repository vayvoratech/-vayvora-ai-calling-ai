import numpy as np
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from redisvl.query import VectorQuery
from sentence_transformers import SentenceTransformer

# 1. Connect to local Redis
redis_client = Redis.from_url("redis://localhost:6379", decode_responses=True)

# 2. Load the SAME Hugging Face model used for insertion
print("📦 Loading transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Connect to your existing index structure
schema_dict = {
    "index": {"name": "local_file_idx", "prefix": "file_chunk:", "storage_type": "hash"},
    "fields": [
        {"name": "text", "type": "text"},
        {
            "name": "vector", 
            "type": "vector", 
            "attrs": {
                "algorithm": "HNSW", 
                "datatype": "FLOAT32", 
                "dims": 384,
                "distance_metric": "COSINE"
            }
        }
    ]
}
index = SearchIndex(IndexSchema.from_dict(schema_dict), redis_client)

# 4. Define the user's question
user_question = "Where is office parking available?"
print(f"\n🔍 Searching for: '{user_question}'")

# 5. Convert the question into a vector and format as bytes
question_vector_list = embedder.encode(user_question).tolist()
question_vector_bytes = np.array(question_vector_list, dtype=np.float32).tobytes()

# 6. Build the Vector Query
query = VectorQuery(
    vector=question_vector_bytes,
    vector_field_name="vector",
    num_results=2,             # Pull top 2 closest matches
    return_fields=["text"]     # Bring back the text field from Redis
)

# 7. Execute the search against Redis
results = index.query(query)

# 8. Print out what was found
print(f"\n📄 Found {len(results)} matching chunks:")
for i, res in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(f"Matched Text: {res['text']}")
    print(f"Distance Score: {res.get('vector_distance', 'N/A')}")