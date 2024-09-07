import asyncio

import logger
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from config import Settings
from services import routers


async def main():
    logger.setup()

    settings = Settings()

    bot = Bot(token=settings.TELEGRAM_BOT_API_TOKEN.get_secret_value())

    storage = MemoryStorage()
    dispatcher = Dispatcher(storage=storage)
    dispatcher.include_routers(*routers)

    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main=main())
