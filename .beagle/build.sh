#!/bin/bash

git config --global --add safe.directory $PWD
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

set -ex
git apply .beagle/s3.patch

if ! [ -e $PWD/.venv/bin/activate ]; then
  python3 -m venv $PWD/.venv
  source $PWD/.venv/bin/activate
fi

make build
