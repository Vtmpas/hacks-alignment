#!/bin/bash

set -ue

sudo docker image prune -af

sudo docker container prune -af
