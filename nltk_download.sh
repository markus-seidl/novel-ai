#!/usr/bin/env bash

source .venv/bin/activate

python -m nltk.downloader -d .venv/nltk_data all
