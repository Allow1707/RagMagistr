from RagMagistr.app.rag.rag_huggingface import rag_huggingface


def init(query: str) -> list:
    res = rag_huggingface(query)
    return res

# from RagMagistr.app.rag.rag_openai import rag_openai
#
#
# def init(query: str) -> list:
#     res = rag_openai(query)
#     return res
