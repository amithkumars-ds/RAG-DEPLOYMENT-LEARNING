from data_ingestion import ingest_pdf
from data_processing import *
from retrieval import *

from sentence_transformers import SentenceTransformer

# loading models and index 
model = SentenceTransformer("models/all-MiniLM-L6-v2")

index = faiss.read_index("vector_store/faiss.index")
with open("vector_store/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"[INFO] model artifacts loaded successfully")

def main():
    reader = ingest_pdf("data/Medical_book.pdf")
    chunks = split_text(reader)
    embeddings, dimension = create_embeddings(chunks, model)
    
    create_index(chunks, embeddings, dimension)

    query = "What is meningitis"
    embedded_query = embed_query(query, model)
    
    fetch_results(embedded_query, index, chunks) 

if __name__ == "__main__":
    main()
