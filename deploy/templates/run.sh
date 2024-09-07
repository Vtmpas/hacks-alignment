#!/bin/bash

set -ue

cd {{ workdir }}

if [ {{ ENVIRON }} = "prod.ini" ]; then
  sudo docker compose up --build --detach --remove-orphans
fi

if [ {{ ENVIRON }} = "test.ini" ]; then
  docker-compose up --build --detach --remove-orphans
fi
