import os
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain_core.embeddings import Embeddings


def convert_data_into_document() -> list:
    # Папка с текстовыми файлами
    DATA_DIR = "data"

    all_splits = []

    # Проходим по всем .txt файлам в папке
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Разбиваем по \n\n (параграфы)
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            # Создаём Document для каждого параграфа
            for i, paragraph in enumerate(paragraphs):
                doc = Document(
                    page_content=paragraph,
                    metadata={"source": filename, "paragraph": i}
                )
                all_splits.append(doc)
    return all_splits


def create_chroma_db(embedding: Embeddings):
    all_splits = convert_data_into_document()

    Chroma.from_documents(
        all_splits,
        embedding=embedding,
        persist_directory="./chroma_db"
    )


if __name__ == '__main__':
    from langchain_huggingface import HuggingFaceEmbeddings
    create_chroma_db(HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))
