#!/bin/bash

set -ue

sudo docker stop deployments-bot-1 deployments-app-1 || true

sudo docker rm -f deployments-bot-1 deployments-app-1 || true
