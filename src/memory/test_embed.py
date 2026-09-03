from redis import Redis
from redisvl.index import SearchIndex
from redisvl.schema import IndexSchema
from sentence_transformers import SentenceTransformer
import numpy as np  # Make sure numpy is imported
# 1. Connect to local Redis
print("🔗 Connecting to Redis...")
redis_client = Redis.from_url("redis://localhost:6379", decode_responses=True)

# 2. Load Local Transformer Model from Hugging Face
# Note: all-MiniLM-L6-v2 outputs 384 dimensions
print("📦 Loading local Hugging Face transformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Define Vector DB Schema (Corrected to use "dims")
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
                "dims": 384,              # <-- Changed from "dim" to "dims"
                "distance_metric": "COSINE"
            }
        }
    ]
}
index = SearchIndex(IndexSchema.from_dict(schema_dict), redis_client)

try:
    index.create(overwrite=True)
    print("✅ Created fresh index in Redis.")
except Exception:
    print("ℹ️ Index already exists, proceeding...")

# 4. Simulate reading a file and breaking it into text chunks
# (You can replace this later with a loop that reads your actual .txt or .pdf files)
raw_file_text = """
Company Policy Overview:
Employees are required to submit their expense reports by the last Friday of every month. 
Late submissions will be processed in the following pay cycle.
For IT support or hardware requests, submit a ticket via the internal portal.
Office parking is available on levels 1 through 3 for registered employee vehicles.
"""

# Simple paragraph/line splitting (Chunking)
chunks = [chunk.strip() for chunk in raw_file_text.split("\n") if chunk.strip()]

print(f"\n📄 Processing {len(chunks)} text chunks with local transformer...")


# 5. Generate local embeddings and pack vectors into bytes for Redis Hashes
data_to_insert = []
for i, text_chunk in enumerate(chunks):
    # Generate the list embedding from Hugging Face
    vector_list = embedder.encode(text_chunk).tolist()
    
    # CONVERT TO BYTES: Required for Redis hash storage type
    vector_bytes = np.array(vector_list, dtype=np.float32).tobytes()
    
    data_to_insert.append({
        "id": str(i),
        "text": text_chunk,
        "vector": vector_bytes  # <-- Now passing bytes instead of a list
    })

# 6. Bulk load the chunks and vectors into Redis Vector DB
index.load(data_to_insert)
print(f"🚀 Successfully inserted {len(data_to_insert)} chunks into Redis using local embeddings!")