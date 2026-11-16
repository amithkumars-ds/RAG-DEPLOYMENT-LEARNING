import faiss
import os, pickle
import numpy as np
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter

# chunking
def split_text(reader:PdfReader) -> list:
    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=150
            )
    chunks = []
    for page in reader.pages:
        chunks_per_page = text_splitter.split_text(page.extract_text())
        chunks.extend(chunks_per_page)
    print(f"Successfully chunked the document")
    print(f"Total number of chunks: {len(chunks)}")

    return chunks

# embedding
def create_embeddings(chunks:list, model) -> np.array:
    embeddings = model.encode(chunks)
    embeddings = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]
    print(f"Dimension of embeddings: {dimension}")
    print(f"Total number of embeddings: {len(embeddings)}")
    
    return embeddings, dimension

# creating an index
def create_index(chunks, embeddings:np.array, dimension:int) -> None:
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    faiss.write_index(index, "vector_store/faiss.index")
    with open("vector_store/chunks.pkl", "wb") as file:
        pickle.dump(chunks, file)
    print("Index created and chunks dumped successfully!")


