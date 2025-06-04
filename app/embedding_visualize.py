import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from langchain.docstore.document import Document
from utils import get_embeddings_model, convert_data_into_documents

# Загружаем документы и эмбеддинги
docs = convert_data_into_documents()
texts = [doc.page_content for doc in docs]
sources = [doc.metadata.get("source", "unknown") for doc in docs]

print(f"Всего чанков: {len(texts)}")
embedding_model = get_embeddings_model("HuggingFace")
embeddings = np.array(embedding_model.embed_documents(texts))

# t-SNE для понижения размерности
print("Понижение размерности...")
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Генерация цветов по источникам
unique_sources = list(sorted(set(sources)))
source_to_color = {src: plt.cm.tab10(i % 10) for i, src in enumerate(unique_sources)}
colors = [source_to_color[src] for src in sources]

# Визуализация
plt.figure(figsize=(10, 8))
for src in unique_sources:
    idxs = [i for i, s in enumerate(sources) if s == src]
    plt.scatter(
        [embeddings_2d[i, 0] for i in idxs],
        [embeddings_2d[i, 1] for i in idxs],
        label=src,
        s=30,
        alpha=0.7
    )

plt.xlabel("TSNE-1")
plt.ylabel("TSNE-2")
plt.legend(loc="best", fontsize=9)
plt.tight_layout()

# Сохраняем
plt.savefig("embedding_space.png", dpi=300)
print("embedding_space.png сохранён!")