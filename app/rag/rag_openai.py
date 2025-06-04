import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()

# Получаем абсолютный путь к директории текущего файла
CURRENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Создаем абсолютный путь к файлу кеша модели
CHROMA_CACHE_PATH = os.path.join(CURRENT_DIR, "chroma_db")


def rag_openai(question: str) -> list[Document]:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    vectorstore = Chroma(
        persist_directory=CHROMA_CACHE_PATH,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large")
    )
    docs = vectorstore.similarity_search_with_relevance_scores(f"query: {question}", k=6)
    content = []
    for text, score in docs:
        print(score)
        if score >= 0.77:
            content.append(text)
    return content
