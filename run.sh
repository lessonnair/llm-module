#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PYTHONPATH=$SCRIPT_DIR

if [ -z "$1" ]; then
  echo "Please provide the task configuration file as an argument"
else
  config_file=$1
  python modules/main.py $config_file
fi