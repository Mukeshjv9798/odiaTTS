#!/bin/bash
apt-get install espeak espeak-ng festival -y
pip install gdown phonemizer torch==1.4.0 soundfile unidecode -y
mkdir ttsmodel
cd ttsmodel
gdown https://drive.google.com/uc?id=1-kJyai2vE2IyLyLfCXw01n-t_bVIS_db
gdown https://drive.google.com/uc?id=13rmXrNFtPYKY2iJZDWFxRo4SzkfzMXBP
cd ..
git clone https://github.com/mukeshjv9798/TTS
cd TTS
python setup.py develop
