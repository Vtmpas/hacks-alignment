#!/bin/bash

set -ue

docker stop hacks-alignment-app-1 hacks-alignment-bot-1 || true

docker rm -f hacks-alignment-app-1 hacks-alignment-bot-1 || true
