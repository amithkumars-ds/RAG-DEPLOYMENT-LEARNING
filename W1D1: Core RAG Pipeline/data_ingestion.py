# data ingestion
from pypdf import PdfReader

def ingest_pdf(filename:str) -> PdfReader:
    reader = PdfReader(filename)
    number_of_pages = len(reader.pages)
    print(f"{filename} loaded successfully!")
    print(f"Total number of pages in book: {number_of_pages}")

    return reader 
    
