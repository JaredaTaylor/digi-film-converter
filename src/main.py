#!/usr/bin/env python3
from __future__ import annotations
from queue import Queue
from threading import Thread
import time
from signal import pause
# from film_emulator import read_image, write_image, emulate_film
from shutter import Shutter

from picamera2 import Picamera2


# def film_worker(queue: Queue):
#     while True:
#         input_path = queue.get()
#         if input_path is None:
#             break

#         try:
#             img = read_image(str(input_path))
#             out = emulate_film(img, preset="portra")

#             output_path = input_path.with_name(f"{input_path.stem}_film{input_path.suffix}")
#             write_image(str(output_path), out)
#             print(f"Processed {output_path}")
#         except Exception as exc:
#             print(f"Film processing failed for {input_path}: {exc}")
#         finally:
#             queue.task_done()


def camera_setup() -> Picamera2:
    picam2 = Picamera2()
    config = picam2.create_still_configuration(main={"size": (4608, 2592)})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    return picam2


def main():
    camera = camera_setup()

    # processing_queue = Queue(maxsize=8)
    # worker = Thread(target=film_worker, args=(processing_queue,), daemon=True)
    # worker.start()

    # shutter = Shutter(camera, processing_queue=processing_queue)
    shutter = Shutter(camera, processing_queue=None)

    shutter.button.when_pressed = shutter.take_photo

    print("Camera ready. Press button to capture.")
    pause()


if __name__ == "__main__":
    main()
