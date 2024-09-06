import asyncio

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from config import Settings


async def main():
    settings = Settings()

    bot = Bot(token=settings.TELEGRAM_BOT_API_TOKEN.get_secret_value())

    storage = MemoryStorage()
    dispatcher = Dispatcher(storage=storage)

    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main=main())
