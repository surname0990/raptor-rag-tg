import logging
import os
from unstructured_ingest import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def constant():
    parent_path = os.getcwd()
    filename = os.path.join(parent_path, "data/fy2024.pdf")
    output_path = os.path.join(parent_path, "data/images")
    return filename, output_path


def main():
    collection_name = "c1"
    logging.info("Started file reader...") 
    raw_pdf_elements = file_reader() 

    logging.info("Number of raw PDF elements: %d", len(raw_pdf_elements))  # Количество прочитанных элементов

    logging.info("text_insert started...")  # Начинаем вставку текста
    text_insert(raw_pdf_elements)

    logging.info("image_insert started...")  # Начинаем вставку изображений
    last_indices = get_last_index_of_page(raw_pdf_elements)  # Получаем последние индексы страниц
    image_insert_with_text(raw_pdf_elements, last_indices)  # Вставляем изображения с текстом
    
    get_docummets()  # Получаем документы

    logging.info("Raptor started...")  # Запускаем Raptor
    raptor_texts = raptor()  # Извлекаем текст с помощью Raptor
    get_documents_with_raptor(raptor_texts)  # Получаем документы с использованием текста от Raptor
    
    logging.info("add data to postgres Started...")  # Начинаем добавление данных в PostgreSQL
    add_docs_to_postgres(collection_name)  # Добавляем документы в базу данных

    logging.info("All Done...")  # Завершаем выполнение

if __name__ == "__main__":
    main()