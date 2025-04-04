from app.rag.rag_huggingface import rag_huggingface
from app.rag.rag_openai import rag_openai


question = "Как разделяется область тонкой структуры в спектрах рентгеновского поглощения (XAFS) и чем отличаются XANES и EXAFS?"

print(rag_huggingface(question))