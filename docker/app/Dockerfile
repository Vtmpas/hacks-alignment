FROM --platform=linux/amd64 nvidia/cuda:12.0.1-base-ubuntu22.04

# Сборка зависимостей
ARG BUILD_DEPS="software-properties-common curl python3.10 python3.10-venv python3.10-dev python3-pip git"
RUN apt-get update && apt-get install -y $BUILD_DEPS

# Установка poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.8.0 POETRY_HOME=/root/poetry python3 -
ENV PATH="${PATH}:/root/poetry/bin"

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Установка питонячьих библиотек
COPY poetry.lock pyproject.toml /
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --without training,dev,bot

# Копирование в контейнер папок и файлов.
COPY app app

ENTRYPOINT ["python3", "app/server.py"]
