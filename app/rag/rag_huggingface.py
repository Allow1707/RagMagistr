from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document


def rag_huggingface(question: str) -> list[Document]:
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma(
        persist_directory="E:\Kods\RagMagistr\\app\chroma_db",
        embedding_function=embedding
    )
    docs: list[Document] = vectorstore.similarity_search(question)
    return docs
