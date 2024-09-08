import asyncio

import logger
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from config import Settings
from services import routers


async def main():
    """
    The main entry point for the Telegram bot application. This function initializes the
    logger, configuration settings, and sets up the bot, dispatcher, and FSM (Finite State Machine) storage.
    It includes routers for handling bot commands and starts polling for incoming updates.

    Steps:
    1. Set up logging using the custom logger.
    2. Load settings from the configuration file (including the Telegram bot token).
    3. Initialize the bot instance using the Telegram bot token.
    4. Set up an in-memory storage for FSM.
    5. Initialize the dispatcher and attach routers for command handling.
    6. Start the bot's long polling to process incoming messages.

    Raises:
        Any exceptions related to bot initialization, dispatcher setup, or polling are
        propagated from the aiogram library.

    Example usage:
        This function is called when the script is executed, and the bot begins polling for updates.
    """
    logger.setup()

    settings = Settings()

    bot = Bot(token=settings.TELEGRAM_BOT_API_TOKEN.get_secret_value())

    storage = MemoryStorage()
    dispatcher = Dispatcher(storage=storage)
    dispatcher.include_routers(*routers)

    await dispatcher.start_polling(bot)


if __name__ == "__main__":
    """
    Entry point for the script. When executed directly, this block runs the main asynchronous
    function using `asyncio.run`, which handles the lifecycle of the bot and dispatcher.

    The `asyncio.run(main())` function ensures that the event loop is started and that the
    bot begins polling for incoming messages.
    """
    asyncio.run(main=main())
