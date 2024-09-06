#!/bin/bash

set -ue

cd {{ workdir }}

docker compose up --build --detach --remove-orphans
