import fitz  # PyMuPDF
import os
from openai import OpenAI
from dotenv import load_dotenv

# Загружаем переменные из .env
load_dotenv()


# --- Функция обращения к модели GPT ---
def model(prompt, user_query):
    # TODO: укажите в файле .env свои  api_key и org_id OpenAi
    open_ai_key: str = os.getenv("OPENAI_API_KEY")
    open_ai_org: str = os.getenv("OPENAI_ORG_ID")

    client: OpenAI = OpenAI(api_key=open_ai_key, organization=open_ai_org)
    os.environ["OPENAI_API_KEY"] = open_ai_key

    QA = client.chat.completions.create(
        model="o4-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query}
        ]
    )

    answer: str = QA.choices[0].message.content
    return answer


# --- Функция для извлечения текста без изображений из PDF ---
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages_text = []

    for page in doc:
        text = page.get_text("text")  # Извлекается только текст
        pages_text.append(text)
    print(pages_text)
    return pages_text


# --- Основной процесс ---
def process_pdf_to_markdown(pdf_path, output_txt_path):
    pages_text = extract_text_from_pdf(pdf_path)
    total_pages = len(pages_text)
    prompt = (
        "Ты — помощник, который форматирует текст научной статьи в Markdown."
        "Игнорируй список литературы, подписи к изображениям. Не добавляй ничего от себя."
        "Не выделяй формулы в отдельный абзац. Вместо этого конкатенируй формулы с абзацем, в котором говориться об этой формуле"
        "Просто отформатируй текст в удобный и читаемый Markdown."
        "Текст в непонятной кодировке по возможности декодируй. Если это невозможно игнорируй его."
    )

    result_markdown = ""

    for i in range(0, total_pages, 2):
        pages_chunk = pages_text[i:i + 2]
        combined_text = "\n\n".join(pages_chunk)

        print(f"Обрабатываю страницы {i + 1} и {i + 2}...")

        markdown = model(prompt, combined_text)
        # result_markdown += f"\n\n<!-- Страницы {i + 1}-{i + 2} -->\n\n"
        result_markdown += markdown
        print(f"Обработка закончена\n")

    # Сохраняем результат
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(result_markdown)

    print(f"\n✅ Готово. Результат сохранён в {output_txt_path}")


# --- Запуск ---
if __name__ == "__main__":
    pdf_path = "..."  # Путь к PDF-файлу который надо распарсить
    output_txt_path = "output.md.txt"  # Куда сохранить результат
    process_pdf_to_markdown(pdf_path, output_txt_path)
