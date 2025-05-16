Вот пример структурированного и читаемого `README.md` файла для твоего проекта RAG-системы:

---

# 🧠 RAG System (Retrieval-Augmented Generation)

Этот проект представляет собой реализацию **RAG-системы (Retrieval-Augmented Generation)** — подхода, сочетающего генеративные языковые модели и поиск по внешней базе знаний. 

В проекте реализовано **две версии RAG**:

- На основе **HuggingFace** моделей
- На основе **OpenAI** (GPT + Embeddings)

---

## 📁 Структура проекта

```bash
project/
│
├── app/               # Основная функционал
│   ├── chroma_db/     # База данных хранящая векторстор
│   │── data/          # Файлы формата .txt которые используются для построения векторной базы
│   │── parsing_pdf/   # Модуль парсинга pdf в текст с помощью нейросети
│       ├── app.py
│   │── rag            # Реализация RAG на HuggingFace и OpenAI 
│       ├── rag_huggingface.py
│       ├── rag_openai.py
│   │── raw_data        # pdf файлы со статьями 
│   │── __init__.py        
│   └──utils.py           # Утилиты для созданию векторной базы
├── utils.py              # Строит векторную базу
├── app.py                # Пример запуска
├── .env                  # Секретные ключи для OpenAI
├── requirements.txt      # Зависимости
└── README.md             # Инструкция по запуску
```

---

## 🚀 Как запустить


### 1. RAG на основе HuggingFace

1. Убедитесь, что в `utils.py` установлен параметр:
   ```python
   embedding_type = "HuggingFace"
   ```
2. Запустите файл:
   ```bash
   python utils.py
   ```
3. Дождитесь сообщения: `Закончили строить векторное хранилище!`

---

### 2. RAG на основе OpenAI

1. Создайте файл `.env` в корне проекта и добавьте туда свои ключи:

   ```
   OPENAI_API_KEY=ваш_ключ
   OPENAI_ORG_ID=ваш_идентификатор_организации
   ```

2. Установите в `utils.py`:
   ```python
   embedding_type = "OpenAI"
   ```

3. Запустите:
   ```bash
   python utils.py
   ```

---

### 4. Интеграция и запуск

- Поместите все файлы из проекта в один каталог вашего проекта.
- В файле `main.py` выберите нужную реализацию функции `init`, в зависимости от типа эмбеддингов.
- Пример использования:

```python
from RagMagistr import main

question = "Что такое RAG?"
result = main.init(question)
print(result)
```

---

## ⚙️ Переменные окружения (.env)

Файл `.env` используется только для OpenAI:

```
OPENAI_API_KEY=your_openai_key
OPENAI_ORG_ID=your_openai_organization
```

---

## 📦 Требуемые библиотеки

Все необходимые зависимости перечислены в `requirements.txt`. Примеры библиотек:

- `langchain==0.3.22`
- `langchain-community==0.3.20`
- `langchain-openai==0.3.11`
- `langchain-huggingface==0.1.2`
- `langchain-chroma==0.2.2`
- `openai==1.70.0`
- `chromadb==0.6.3`
- `PyMuPDF==1.25.5`
- `python-dotenv==1.1.0`

---

