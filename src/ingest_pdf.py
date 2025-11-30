import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vectordb import VectorDB
import os
from dotenv import load_dotenv

load_dotenv()

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_with_metadata(text: str, source: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_text(text)
    
    # Add page/section metadata (approximate)
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append({
            "content": chunk,
            "metadata": {
                "source": source,
                "chunk_id": i,
                "type": "cardiovascular_pharmacotherapy"
            }
        })
    return docs

def main():
    pdf_path = "data/cardiovascular_pharmacotherapy.pdf"
    print(f"Extracting text from {pdf_path}...")
    full_text = extract_text_from_pdf(pdf_path)
    
    print(f"Extracted {len(full_text):,} characters. Chunking...")
    documents = chunk_with_metadata(full_text, pdf_path)
    
    print(f"Created {len(documents)} chunks. Adding to vector DB...")
    db = VectorDB(collection_name="cv_pharmacotherapy")
    db.add_documents(documents)
    
    print("Ingestion complete! Ready for Q&A.")

if __name__ == "__main__":
    main()