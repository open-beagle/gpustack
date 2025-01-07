#!/bin/bash

git config --global --add safe.directory $PWD
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/

set -ex

if ! [ -e $PWD/.venv/bin/activate ]; then
  python3 -m venv $PWD/.venv
  source $PWD/.venv/bin/activate
fi

if $(git diff --quiet pyproject.toml); then
  git apply .beagle/v0.4.1-s3-project.patch
fi

if $(git diff --quiet gpustack/worker/backends/base.py); then
  git apply .beagle/v0.4.1-s3.patch
fi

if $(git diff --quiet gpustack/cmd/download_tools.py); then
  git apply .beagle/v0.4.1-logginglocal.patch
fi

make build

git apply -R .beagle/v0.4.1-s3-project.patch
git apply -R .beagle/v0.4.1-s3.patch
git apply -R .beagle/v0.4.1-logginglocal.patch
