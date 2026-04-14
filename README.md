# digi-film-converter
Python utility for digital film camera project.

# Setup
## install system deps (Pi only)
sudo apt update
sudo apt install -y python3-picamera2 python3-libcamera

## create venv WITH system packages
python3 -m venv venv --system-site-packages

## activate
source venv/bin/activate

## install python deps
pip install --upgrade pip
pip install -r requirements.txt