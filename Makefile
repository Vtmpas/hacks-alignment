# Базовый Makefile для Python библиотеки с использованием Poetry

# Переменные
POETRY := poetry
PYTHON := python

.PHONY: install update test install-test-deps lint format clean environment build deploy destroy restart

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

build:
	docker login -u konductor14 -p $(DOCKER_REGISTRY_PASSWORD)
	docker build -f docker/app/Dockerfile -t konductor14/hacks-alignment-app:latest .
	docker build -f docker/bot/Dockerfile -t konductor14/hacks-alignment-bot:latest .
	docker push konductor14/hacks-alignment-app:latest
	docker push konductor14/hacks-alignment-bot:latest

# Сервер
INVENTORY=
ifeq ($(evn), test)
	INVENTORY=deploy/test.ini
else ifeq ($(evn), prod)
	INVENTORY=deploy/prod.ini
else
	INVENTORY=deploy/test.ini
endif

deploy:
	@echo "Using $(INVENTORY)"
	ansible-playbook -i $(INVENTORY) deploy/deploy.yml -e ENVIRON=$(INVENTORY)

destroy:
	@echo "Using $(INVENTORY)"
	ansible-playbook -i $(INVENTORY) deploy/destroy.yml

restart:
	@echo "Using $(INVENTORY)"
	ansible-playbook -i $(INVENTORY) deploy/restart.yml

%:
	@if [ "$(origin CMD)" = "undefined" ]; then \
		$(error Unknown command. Use 'deploy' or 'destroy' or 'restart'); \
	fi
	@$(MAKE) env=$@ $(CMD)
