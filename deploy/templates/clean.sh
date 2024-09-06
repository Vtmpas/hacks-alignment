#!/bin/bash

set -ue

docker image prune -af

docker container prune -af
