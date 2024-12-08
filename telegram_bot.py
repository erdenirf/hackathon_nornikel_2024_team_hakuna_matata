import asyncio
import logging
import PIL
from aiogram import F, Bot, Dispatcher, types
from aiogram.filters.command import Command
from config_reader import config
from src.ColQwen2Embeddings import ColQwen2Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from langchain.retrievers.ensemble import EnsembleRetriever
from aiogram.utils.formatting import (
    Bold, as_list, as_marked_section, HashTag
)
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI
import zlib
from pathlib import Path
from src.NornikelPdfLoader import NornikelPdfLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)

client = OpenAI(
    base_url=config.LLM_BASE_URL.get_secret_value(),
    api_key=config.LLM_API_KEY.get_secret_value(),
)

logging.info("Loading embeddings")
embeddings = ColQwen2Embeddings()
logging.info("Embeddings loaded")

DB_LIST: dict = {}

qdrant = QdrantVectorStore.from_documents(
    documents=[],
    embedding=embeddings,
    url=config.QDRANT_URL.get_secret_value(),
    api_key=config.QDRANT_API_KEY.get_secret_value(),
    collection_name=config.QDRANT_COLLECTION_NAME.get_secret_value(),
    force_recreate=True
)

retriever_text = qdrant.as_retriever(search_type="mmr", 
                                search_kwargs={"k": 7,
    "filter": models.Filter(
        should=[
            models.FieldCondition(
                key="metadata.type",
                match=models.MatchValue(
                    value="text"
                ),
            ),
        ]
    )
})

retriever_image = qdrant.as_retriever(search_type="mmr", 
                                search_kwargs={"k": 3,
    "filter": models.Filter(
        should=[
            models.FieldCondition(
                key="metadata.type",
                match=models.MatchValue(
                    value="image"
                ),
            ),
        ]
    )
})

ensemble_retriever = EnsembleRetriever(retrievers=[retriever_text, retriever_image],
                                      weights=[0.5, 0.5])

# Объект бота
bot = Bot(token=config.BOT_TOKEN.get_secret_value())
# Диспетчер
dp = Dispatcher()

# Создаем список с командами и их описанием
commands = [
    types.BotCommand(command="start", description="Начать работу с ботом"),
    types.BotCommand(command="list_indexed", description="Получить список индексированных документов"),
    types.BotCommand(command="upload_pdf", description="Загрузить PDF документ"),
    types.BotCommand(command="del_indexed", description="Удалить индексированный документ"),
]

# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Я бот для работы с PDF документами.\n"
                        "Используйте команды:\n"
                        "/upload_pdf - для загрузки документа\n"
                        "/list_indexed - для просмотра списка документов\n"
                        "/del_indexed - для удаления документа.\n\n"
                        "Отправьте поисковое сообщение для получения ответа от ИИ с учетом контекста документа.")

# Хэндлер на команду /list_indexed
@dp.message(Command("list_indexed"))
async def cmd_list_indexed(message: types.Message):
    files: list[str] = sorted(list(DB_LIST.values())) if len(DB_LIST) > 0 else ["Список пуст"]
    await message.answer("Список проиндексированных документов:\n"
                        "\n".join(files))

# Хэндлер на команду /upload_pdf
@dp.message(Command("upload_pdf"))
async def cmd_upload_pdf(message: types.Message):
    await message.answer("Пожалуйста, загрузите PDF-файл как документ.")

# Хэндлер на получение PDF файла
@dp.message(F.document)
async def handle_pdf_document(message: types.Message):
    if message.document.mime_type == "application/pdf":
        # Download the file
        file = await bot.get_file(message.document.file_id)
        file_content = await bot.download_file(file.file_path)
        
        # Calculate CRC32 checksum
        content_bytes = file_content.read()
        checksum = zlib.crc32(content_bytes)
        CRC32 = hex(checksum)

        if CRC32 in DB_LIST:
            await message.answer(f"Этот файл уже был загружен ранее.")
        else:
            await message.answer(f"Файл загружен и будет проиндексирован. Ожидайте...")
            DB_LIST[CRC32] = message.document.file_name
            
            # Save the file temporarily to disk
            temp_file_path = message.document.file_name
            with open(temp_file_path, "wb") as temp_file:
                file_content.seek(0)  # Reset file pointer
                temp_file.write(content_bytes)
            
            try:
                # Use the temporary file path with the loader
                loader = NornikelPdfLoader(temp_file_path)
                docs = loader.load()
                splitted_docs = text_splitter.split_documents(docs)
                for doc in splitted_docs:
                    qdrant.add_documents([doc])
                    
                await message.answer(
                    f"Файл успешно проиндексирован!\n"
                    f"Получен PDF файл: {message.document.file_name}"
                )
            finally:
                # Clean up the temporary file
                import os
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
    else:
        await message.answer("Пожалуйста, отправьте файл в формате PDF")

# Хэндлер на команду /del_indexed
@dp.message(Command("del_indexed"))
async def cmd_del_indexed(message: types.Message):
    await message.answer("Введите номер или название документа для удаления\n"
                        "(Здесь будет логика удаления)")

# Заменяем общий хэндлер на текстовые сообщения
@dp.message(F.text)
async def handle_text(message: types.Message):
    #results = ensemble_retriever.invoke(message.text)
    results_text = retriever_text.invoke(message.text)
    results_image = retriever_image.invoke(message.text)

    texts = [f"{result.metadata['source']}/{result.metadata['page']} стр./{result.page_content}" for result in results_text if result.metadata['type'] == 'text']
    images = [result.metadata['image_base64'] for result in results_image if result.metadata['type'] == 'image']
    images_captions = [f"{result.metadata['source']}/{result.metadata['page']} стр." for result in results_image if result.metadata['type'] == 'image']
    
    for index, image in enumerate(images):
        try:
            # Логируем начало строки base64 для отладки
            logging.debug(f"Base64 string preview: {image[:50]}...")
            
            # Очистка строки base64 от возможного префикса данных
            if ',' in image:
                image = image.split(',')[1]
            
            # Добавляем padding если необходимо
            padding = 4 - (len(image) % 4) if len(image) % 4 else 0
            image = image + ('=' * padding)
            
            # Декодируем base64 в бинарные данные
            img_data = base64.b64decode(image)
            
            # Проверяем на пустые данные
            if len(img_data) == 0:
                logging.error("Получены пустые данные изображения")
                continue
            
            # Создаем объект BytesIO для работы с бинарными данными
            bio = BytesIO(img_data)
            bio.seek(0)
            
            # Открываем и обрабатываем изображение
            img = Image.open(bio)
            img = img.convert('RGB')  # Конвертируем в RGB формат
            
            # Сохраняем обработанное изображение
            output_bio = BytesIO()
            img.save(output_bio, format='JPEG', quality=95)
            output_bio.seek(0)
            
            # Отправляем изображение в чат
            await message.answer_photo(
                types.BufferedInputFile(
                    output_bio.getvalue(),
                    filename="image.jpg"
                ),
                caption=images_captions[index]
            )
            
        except base64.binascii.Error as e:
            logging.error(f"Ошибка декодирования base64: {e}", exc_info=True)
            continue
        except PIL.UnidentifiedImageError as e:
            logging.error(f"Неподдерживаемый формат изображения: {e}", exc_info=True)
            continue
        except Exception as e:
            logging.error(f"Ошибка при обработке изображения: {e}", exc_info=True)
            continue

    for index, text in enumerate(texts):
        content = as_list(
            as_marked_section(
                Bold(f"Multi-modal RAG context [{index+1}]:"),
                text,
                marker="🔎 ",
            ),
            sep="\n\n",
        )
        await message.answer(**content.as_kwargs())

    rag_context = "\n\n".join(texts)
    image_urls = []
    for image in images:
        # Clean up base64 string if it has data URI prefix
        if ',' in image:
            image = image.split(',')[1]
        # Add proper base64 padding
        padding = 4 - (len(image) % 4) if len(image) % 4 else 0
        image = image + ('=' * padding)
        # Format as proper base64 data URI
        image_data = f"data:image/jpeg;base64,{image}"
        image_urls.append({
            "type": "text",  # Changed from "image_url" to "text"
            "text": image_data  # Changed from "image_url" to "text"
        })

    completion = client.chat.completions.create(
        model="Qwen/Qwen2-VL-72B-Instruct-AWQ",
        messages=[
            {"role": "system", "content": """Я - виртуальный ассистент компании «Норникель», созданный для помощи с документами и информацией компании. Я использую мультимодальную RAG-систему для поиска и анализа релевантной информации в корпоративных документах, изображениях и других материалах.

        Я отвечаю на вопросы, основываясь на предоставленном контексте из базы знаний.

        Мои ответы всегда вежливы, профессиональны и соответствуют корпоративной культуре Норникеля."""},
            {"role": "system", "content": rag_context},
            {"role": "user", "content": [
                {"type": "text", "text": message.text},
                # *image_urls
            ]}
        ])

    content = as_list(
            as_marked_section(
                Bold(f"LLM's answer (Qwen2-VL-72B-Instruct-AWQ):"),
                completion.choices[0].message.content,
                marker="🤖 ",
            ),
            sep="\n\n",
        )
    await message.answer(**content.as_kwargs())

# Запуск процесса поллинга новых апдейтов
async def main():
    # Установка команд бота
    await bot.set_my_commands(commands)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())