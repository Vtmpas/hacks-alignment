# Базовый Makefile для Python библиотеки с использованием Poetry

# Переменные
POETRY := poetry
PYTHON := python

.PHONY: install update test install-test-deps lint format clean environment build

install:
	$(POETRY) install

update:
	$(POETRY) update

test: install-test-deps
	$(POETRY) run pytest

install-test-deps:
	$(POETRY) install --no-interaction --no-ansi --only dev --all-extras

lint:
	$(POETRY) run ruff check .
	$(POETRY) run mypy .

format:
	$(POETRY) run ruff format
	$(POETRY) run ruff check --fix .

clean:
	# Удаление файлов .pyc, .pyo, .pyd
	find . -type f -name "*.py[cod]" -exec rm -f {} +
	# Удаление директорий сборки
	rm -rf build dist *.egg-info
	# Удаление кэшей mypy и pytest
	rm -rf .mypy_cache/ .pytest_cache/
	# Удаление файлов покрытия кода и HTML отчета
	rm -rf .coverage coverage.xml htmlcov/
	# Удаление файлов логов
	find . -type f -name "*.log" -exec rm -f {} +

environment:
	$(POETRY) export -f requirements.txt --output requirements.txt --without dev --without-hashes

build:
	docker build -t hacks-alignment .
