import faiss
import pickle

# query embedding
def embed_query(query:str, model):
    embedded_query = model.encode([query]).astype("float32")
    return embedded_query

# retrieval
def fetch_results(embedded_query, index, chunks):
    faiss.normalize_L2(embedded_query)
    distances, indices = index.search(embedded_query, k=5)

    # results
    print("\nRetrieval Results")
    print("-"*50)
    for idx, distance in zip(indices[0], distances[0]):
        print(f"Distance: {distance}")
        print(chunks[idx])
        print("-"*50)
