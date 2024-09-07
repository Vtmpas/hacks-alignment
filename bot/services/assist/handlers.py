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
    await state.set_state(state=AssistUserStates.free)
    await message.answer(text=WELCOME_MSG)


@router.message(AssistUserStates.free)
async def handler_free(message: Message, state: FSMContext) -> None:
    async def app_service_request(query: str) -> str:
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
    await message.answer(text=BUSY_MSG)
