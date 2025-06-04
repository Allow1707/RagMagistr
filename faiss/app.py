import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Загружаем переменные окружения (например, OPENAI_API_KEY)
os.environ["OPENAI_API_KEY"] = "..."

# Инициализируем эмбеддинги
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Папка с текстовыми файлами
DATA_DIR = "data"

def load_and_split_documents(data_dir: str) -> list[Document]:
    documents = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as f:
                text = f.read()

            # Создаем Document
            documents.append(Document(
                page_content=text,
                metadata={"source": filename}
            ))

    # Разбиваем на чанки
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", " "]
    )
    split_docs = text_splitter.split_documents(documents)
    return split_docs

def build_vectorstore(docs: list[Document]) -> Chroma:
    vectorstore = Chroma.from_documents(docs,embedding=embedding_model,
        collection_metadata={"hnsw:space": "l2"})
    return vectorstore

def main():
    print("Загружаем и обрабатываем документы...")
    docs = load_and_split_documents(DATA_DIR)

    print(f"Количество чанков: {len(docs)}")

    # Построение векторного хранилища
    vectorstore = build_vectorstore(docs)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # Пример запроса
    while True:
        query = input("Введите запрос (или 'exit'): ").strip()
        if query.lower() == "exit":
            break

        results = retriever.get_relevant_documents(query)

        print("\n🔍 Найденные документы:\n")
        for i, doc in enumerate(results):
            print(f"--- Результат {i+1} ---")
            print(doc.page_content)
            print(f"Источник: {doc.metadata.get('source')}")
            print()

        # Проверка на точное совпадение
        print("🔎 Поиск точного вхождения цитаты в базе...")
        found = False
        for doc in docs:
            if query in doc.page_content:
                print("✅ Найдено точное совпадение!")
                print(doc.page_content)
                print(f"Источник: {doc.metadata}")
                found = True
                break
        if not found:
            print("❌ Точного совпадения не найдено.")

if __name__ == "__main__":
    main()