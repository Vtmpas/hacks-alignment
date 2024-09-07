#!/bin/bash

set -ue

cd {{ workdir }}

sudo docker compose restart
