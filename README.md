# digi-film-converter
Python utility for digital film camera project.

# Setup
## install system deps (Pi only)
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera

## install desktop tuner UI deps
sudo apt install -y python3-tk tk

## create venv WITH system packages
python3 -m venv venv --system-site-packages

## activate
source venv/bin/activate

## install python deps
pip install --upgrade pip
pip install -r requirements.txt
