import asyncio

import aiohttp
from aiogram import Router
from aiogram.filters.command import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from .phrases import BUSY_MSG, WELCOME_MSG
from .states import AssistUserStates

router = Router(name="assist")


@router.message(CommandStart())
async def welcome(message: Message, state: FSMContext) -> None:
    """
    Handles the /start command from the user. Sets the user's state to `free` and sends
    a welcome message.

    Args:
        message (Message): The incoming message object containing the /start command.
        state (FSMContext): The finite state machine context for managing user states.

    Returns:
        None
    """
    await state.set_state(state=AssistUserStates.free)
    await message.answer(text=WELCOME_MSG)


@router.message(AssistUserStates.free)
async def handler_free(message: Message, state: FSMContext) -> None:
    """
    Handles messages from users in the `free` state. Processes the message by making
    an asynchronous request to an external service and sends back the response.

    Args:
        message (Message): The incoming message object containing the user's query.
        state (FSMContext): The finite state machine context for managing user states.

    Returns:
        None
    """

    async def app_service_request(query: str) -> str:
        """
        Makes an asynchronous HTTP POST request to an external service with the provided query.

        Args:
            query (str): The query string to send in the POST request.

        Returns:
            str: The response text from the external service.
        """
        timeout = aiohttp.ClientTimeout(
            total=None,
            connect=10,
            sock_connect=10,
            sock_read=60,
        )

        url = "http://app:8000/assist"
        json = {"query": query}

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url=url, json=json) as response:
                return await response.text()

    await state.set_state(state=AssistUserStates.busy)

    response_event = asyncio.Event()

    try:
        response = await asyncio.wait_for(app_service_request(query=(message.text or "").strip()), timeout=60)
        response_event.set()

    except asyncio.TimeoutError:
        response = "Request timed out. Try to send new one."

    except Exception as e:
        response = f"Error while processing request: {str(e)}"

    finally:
        await state.set_state(state=AssistUserStates.free)

    await message.answer(text=response)


@router.message(AssistUserStates.busy)
async def handler_busy(message: Message) -> None:
    """
    Handles messages from users in the `busy` state. Sends a message indicating that
    the system is currently busy and cannot process additional requests.

    Args:
        message (Message): The incoming message object.

    Returns:
        None
    """
    await message.answer(text=BUSY_MSG)
