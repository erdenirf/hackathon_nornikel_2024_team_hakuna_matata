import asyncio
import logging
from aiogram import F, Bot, Dispatcher, types
from aiogram.filters.command import Command
from config_reader import config
from src.ColQwen2Embeddings import ColQwen2Embeddings
from langchain_qdrant import QdrantVectorStore

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)

logging.info("Loading embeddings")
embeddings = ColQwen2Embeddings()
logging.info("Embeddings loaded")

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=config.QDRANT_COLLECTION_NAME.get_secret_value(),
    url=config.QDRANT_URL.get_secret_value(),
    api_key=config.QDRANT_API_KEY.get_secret_value()
)

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
                        "/del_indexed - для удаления документа")

# Хэндлер на команду /list_indexed
@dp.message(Command("list_indexed"))
async def cmd_list_indexed(message: types.Message):
    # Здесь будет логика получения списка документов
    await message.answer("Список индексированных документов:\n"
                        "1. Document1.pdf\n"
                        "2. Document2.pdf\n"
                        "(Это тестовый список)")

# Хэндлер на команду /upload_pdf
@dp.message(Command("upload_pdf"))
async def cmd_upload_pdf(message: types.Message):
    await message.answer("Пожалуйста, отправьте PDF файл для загрузки")

# Хэндлер на получение PDF файла
@dp.message(F.document)
async def handle_pdf_document(message: types.Message):
    if message.document.mime_type == "application/pdf":
        await message.answer(f"Получен PDF файл: {message.document.file_name}\n"
                           f"(Здесь будет логика обработки файла)")
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
    await message.answer("Используйте команды бота для взаимодействия.\n"
                        "Отправьте /start для получения списка команд.")

# Запуск процесса поллинга новых апдейтов
async def main():
    # Установка команд бота
    await bot.set_my_commands(commands)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())