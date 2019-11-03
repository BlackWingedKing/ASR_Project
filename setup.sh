#!/bin/bash

mkdir -p data
mkdir -p data/train data/temp
mkdir -p data/train/full_vid data/train/shifted data/train/unshifted

pip install -r requirements.txt

# sudo apt-get install ffmpeg sox
