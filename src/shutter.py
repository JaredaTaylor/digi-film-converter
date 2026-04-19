#!/usr/bin/env python3
from gpiozero import Button
from picamera2 import Picamera2
from pathlib import Path
from datetime import datetime
from queue import Queue
import time

SAVE_DIR = Path.home() / "photos"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

class Shutter:
    def __init__(self, camera: Picamera2, processing_queue: Queue | None, pin=17, pull_up=True, bounce_time=0.1):
        self.camera = camera
        self.button = Button(pin, pull_up=pull_up, bounce_time=bounce_time)
        self.button.when_pressed = self.take_photo
        self.last_capture_time = 0
        self.capture_cooldown = 1.0
        self.processing_queue = processing_queue

    def take_photo(self):
        now = time.time()
        if now - self.last_capture_time < self.capture_cooldown:
            return

        self.last_capture_time = now

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = SAVE_DIR / f"image_{timestamp}.jpg"

        print(f"Capturing {filename} ...")
        self.camera.capture_file(str(filename))
        if self.processing_queue:
            self.processing_queue.put_nowait(filename)
        print("Saved.")
