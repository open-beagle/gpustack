#!/bin/bash

git config --global --add safe.directory $PWD
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

set -ex

if ! [ -e $PWD/.venv/bin/activate ]; then
  python3 -m venv $PWD/.venv
  source $PWD/.venv/bin/activate
fi

if $(git diff --quiet gpustack/logginglocal.py); then
  git apply .beagle/v0.4.1-logginglocal.patch
fi

make build

git apply -R .beagle/v0.4.1-logginglocal.patch
