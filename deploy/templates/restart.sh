#!/bin/bash

set -ue

cd {{ workdir }}

docker composeя restart
