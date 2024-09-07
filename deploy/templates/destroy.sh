#!/bin/bash

set -ue

docker stop deployments-bot-1 deployments-app-1 || true

docker rm -f deployments-bot-1 deployments-app-1 || true
