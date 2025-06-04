import requests
import os
import xml.etree.ElementTree as ET


def process_pdf_with_grobid(pdf_path, grobid_url="http://localhost:8070/api/processFulltextDocument"):
    # Проверяем существование файла
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF файл не найден: {pdf_path}")

    # Проверяем размер файла
    file_size = os.path.getsize(pdf_path)
    print(f"Размер файла: {file_size / (1024 * 1024):.2f} MB")

    # Добавляем таймаут и заголовки
    headers = {
        'Accept': 'application/xml'
    }

    try:
        with open(pdf_path, 'rb') as pdf_file:
            files = {
                'input': (os.path.basename(pdf_path), pdf_file, 'application/pdf')
            }

            # Увеличиваем таймаут для больших файлов
            response = requests.post(
                grobid_url,
                files=files,
                headers=headers,
                timeout=300  # 5 минут таймаут
            )

            if response.status_code == 200:
                return response.text
            else:
                # Более подробная информация об ошибке
                error_msg = f"""
                Ошибка от GROBID:
                Статус код: {response.status_code}
                Ответ: {response.text}
                URL: {grobid_url}
                """
                raise Exception(error_msg)

    except requests.exceptions.ConnectionError:
        raise Exception("Не удалось подключиться к GROBID. Убедитесь, что сервис запущен.")
    except requests.exceptions.Timeout:
        raise Exception("Превышено время ожидания ответа от GROBID.")
    except Exception as e:
        raise Exception(f"Неожиданная ошибка при обработке PDF: {str(e)}")



def remove_namespace(tree):
    for elem in tree.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    return tree

def extract_divs_from_tei(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    remove_namespace(tree)
    body = root.find('.//text/body')

    final_text = ""
    if body is None:
        print("Раздел <body> не найден.")
        return
    divs = body.findall('div')
    print(f"Найдено {len(divs)} <div> блоков.\n")
    for i, div in enumerate(divs, 1):
        page = ""
        text = ""
        print(f"--- DIV #{i} ---")
        head = div.find('head')
        if head is not None and head.text:
            print(f"[Заголовок]: {head.text.strip()}")
        for p in div.findall('p'):
            if p.text:
                text += p.text.strip()

        if text:
            if head is not None and head.text:
                page = f"{head.text.strip()}:\n{text}\n\n"
            else:
                page = f"{text}\n\n"
        final_text += page
    with open(f"{BOOK_NAME}.txt", "w", encoding="utf-8") as f:
        f.write(final_text)

if __name__ == "__main__":
    BOOK_NAME = "XAFS+Techniques+for+Catalysts%2C+Nanomater.compressed"

    xml_file = f"{BOOK_NAME}.xml"
    tei_result = process_pdf_with_grobid(f"{BOOK_NAME}.pdf")
    with open(xml_file, "w", encoding="utf-8") as f:
        f.write(tei_result)
    extract_divs_from_tei(xml_file)