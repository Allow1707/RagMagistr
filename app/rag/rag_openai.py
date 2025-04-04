import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document


def rag_openai(question: str) -> list[Document]:
    os.environ["OPENAI_API_KEY"] = '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'
    with open("../data/применение XAS в материаловедении.txt", "r", encoding="utf-8") as f:
        text = f.read()

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    all_splits = [Document(page_content=p) for p in paragraphs]

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    docs = vectorstore.similarity_search(question)
    return docs
