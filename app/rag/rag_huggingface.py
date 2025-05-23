import os
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from app.utils import get_embeddings_model, timing_decorator

# Получаем абсолютный путь к директории текущего файла
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Создаем абсолютный путь к файлу кеша модели
CHROMA_CACHE_PATH = os.path.join(CURRENT_DIR, "chroma_db")


@timing_decorator
def rag_huggingface(question: str) -> list:
    embedding = get_embeddings_model()
    vectorstore = Chroma(
        persist_directory=CHROMA_CACHE_PATH,
        embedding_function=embedding
    )
    docs = vectorstore.similarity_search_with_relevance_scores(question, k=6)
    content = []
    for text, score in docs:
        print(score)
        if score >= 0.77:
            content.append(text)
    return content
