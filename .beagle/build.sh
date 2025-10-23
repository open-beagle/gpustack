#!/bin/bash

git config --global --add safe.directory "$PWD"
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

set -ex

if ! [ -e "$PWD"/.venv/bin/activate ]; then
  python3 -m venv "$PWD"/.venv
  source "$PWD"/.venv/bin/activate
fi

if git diff --quiet pyproject.toml; then
  git apply .beagle/v0.7.0-s3-project.patch
fi

make VERSION=v0.7.0 build

git apply -R .beagle/v0.7.0-s3-project.patch