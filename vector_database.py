import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings


pdfs_directory = "pdfs/"
os.makedirs(pdfs_directory, exist_ok=True)

def upload_pdf(file):
    with open(os.path.join(pdfs_directory, file.name), "wb") as f:
        f.write(file.getbuffer())


def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()
    return documents

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_faiss_db_from_pdf(file) -> FAISS:
    print("[INFO] Uploading file...")
    upload_pdf(file)

    file_path = os.path.join(pdfs_directory, file.name)
    print(f"[INFO] File saved to: {file_path}")

    documents = load_pdf(file_path)
    print(f"[INFO] Loaded {len(documents)} documents.")

    if not documents:
        raise ValueError("No readable content found in the uploaded PDF.")

    text_chunks = create_chunks(documents)
    print(f"[INFO] Created {len(text_chunks)} chunks.")

    embeddings = get_embedding_model()
    print("[INFO] Embedding model loaded.")

   
    test_vector = embeddings.embed_query("Hello World")
    print(f"[DEBUG] Test vector length: {len(test_vector)}")

    faiss_db = FAISS.from_documents(text_chunks, embeddings)
    print("[INFO] FAISS index created successfully.")

    return faiss_db
