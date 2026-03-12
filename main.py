from src.file_uploader.upload_document import upload_document
from langchain_huggingface import HuggingFaceEmbeddings
from src.db_setup.initialize_DB import initialize_DB
import torch

collection = initialize_DB(True)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device},
)

def main():

    file_path = upload_document()

    if not file_path:
        return

    docs = load_pdf(file_path)

    text = "\n".join([doc.page_content for doc in docs])

    chunks = chunk_document(text)

    embeddings = embed_document(chunks, model)

    store_vectors(chunks, embeddings, collection, file_path.name)


if __name__ == "__main__":
    main()
