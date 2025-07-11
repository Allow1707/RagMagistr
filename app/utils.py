import os
import logging
import pickle
import shutil
import time
import functools
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from dotenv import load_dotenv

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Загружаем переменные из .env
load_dotenv()

# Получаем абсолютный путь к директории текущего файла
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Создаем абсолютный путь к файлу кеша модели
MODEL_CACHE_PATH = os.path.join(CURRENT_DIR, "embeddings_model.pkl")


def timing_decorator(func):
    """Декоратор для измерения времени выполнения функции"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        print(f"Начинаем выполнение: {func.__name__}...")
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Завершено: {func.__name__} за {end - start:.2f} s\n")
        return result

    return wrapper


@timing_decorator
def get_embeddings_model(embedding_type: str = "HuggingFace"):
    """
    Загружает модель из кэша или инициализирует новую
    @embedding_type: enum. {HuggingFace, OpenAI}
    """

    if embedding_type == "HuggingFace":
        if os.path.exists(MODEL_CACHE_PATH):
            print("Загрузка модели из кэша...")
            try:
                with open(MODEL_CACHE_PATH, "rb") as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Ошибка загрузки модели из кэша: {e}")
                # Если не удалось загрузить, создаем новую

        print("Инициализация новой модели...")
        model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

        # Сохраняем модель в кэш
        try:
            with open(MODEL_CACHE_PATH, "wb") as f:
                pickle.dump(model, f)
            print("Модель сохранена в кэш")
        except Exception as e:
            print(f"Ошибка сохранения модели в кэш: {e}")
    elif embedding_type == "OpenAI":
        os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
        model = OpenAIEmbeddings()
    else:
        raise Exception(f"Model {embedding_type} doesn't allowed")

    return model


def convert_data_into_documents(max_tokens: int = 1024) -> list[Document]:
    """
    Загружает .txt-файлы, разбивает их на чанки и добавляет префикс 'passage:',
    при этом выводит количество токенов в каждом чанке.
    Можно указать лимит `max_tokens` — чанки с превышением будут отфильтрованы.
    """
    DATA_DIR = "data"
    documents = []

    # Инициализация токенизатора от модели
    tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")

    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
                text = f.read()

            documents.append(Document(
                page_content=text,
                metadata={"source": filename}
            ))

    # Разделение на чанки
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=250,
        separators=["\n\n", ".", "\n", " "]
    )
    split_docs = text_splitter.split_documents(documents)

    final_docs = []
    for idx, doc in enumerate(split_docs):
        text_with_prefix = f"passage: {doc.page_content.strip()}"
        num_tokens = len(tokenizer.tokenize(text_with_prefix))

        print(f"Чанк #{idx+1}: {num_tokens} токенов")
        if num_tokens <= max_tokens:
            doc.page_content = text_with_prefix
            final_docs.append(doc)
        else:
            print(f"⚠️ Пропущен — превышает {max_tokens} токенов")

    print(f"\nИтого: {len(final_docs)} чанков использовано (лимит: {max_tokens} токенов)")
    return final_docs


def clean_chroma_directory():
    """Удаляет все файлы из директории ./chroma_db"""
    chroma_dir = "./chroma_db"
    if os.path.exists(chroma_dir):
        print(f"Удаляем старое векторное хранилище из {chroma_dir}...")
        try:
            shutil.rmtree(chroma_dir)
            print(f"Директория {chroma_dir} успешно очищена")
        except Exception as e:
            print(f"Ошибка при удалении директории {chroma_dir}: {e}")
    else:
        print(f"Директория {chroma_dir} не существует, создаем новую")

    # Убедимся, что директория существует перед созданием хранилища
    os.makedirs(chroma_dir, exist_ok=True)


@timing_decorator
def create_chroma_db(embedding: Embeddings):
    all_splits = convert_data_into_documents()

    # Очищаем директорию перед созданием нового хранилища
    clean_chroma_directory()
    print("Cтроим векторное хранилище")
    Chroma.from_documents(
        all_splits,
        embedding=embedding,
        persist_directory="./chroma_db",
        collection_metadata={"hnsw:space": "l2"}
    )
    print("Закончили строить векторное хранилище!")


if __name__ == '__main__':
    embedding_type = "HuggingFace"
    embeddings_model = get_embeddings_model(embedding_type)
    create_chroma_db(embeddings_model)
