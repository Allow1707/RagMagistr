from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def is_chroma_db_empty(persist_directory: str = "./chroma_db") -> bool:
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Загружаем векторное хранилище
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )

    # Получаем количество документов
    try:
        collection = vectorstore._collection  # внутренний объект Chroma
        count = collection.count()
        print(f"Документов в базе: {count}")
        return count == 0
    except Exception as e:
        print(f"Ошибка при проверке базы: {e}")
        return True  # безопасно считаем, что база пуста

# Пример использования
if is_chroma_db_empty("./chroma_db"):
    print("База пуста или не найдена.")
else:
    print("База содержит документы.")