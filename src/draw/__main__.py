import argparse
import os
import shutil
import sys
import time
import urllib
from functools import wraps
from pathlib import Path
from typing import Callable, Any

import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

from draw.draw import Draw
from draw.logger import DetectionLogger

OpenCVImage = cv2.Mat | np.ndarray[Any, np.dtype]


def parse_command_line():
    parser = argparse.ArgumentParser('Detecting and Recognizing a Wide Range of Yu-Gi-Oh! Cards', add_help=True)

    parser.add_argument('--source', default='0', type=str,
                        help="Source to use for processing refer to github.com/HichTala/draw2 for more details")

    parser.add_argument("--save", nargs="?", const="output",
                        help="Save the video. Optionally provide a path. If not, saves as 'output' in current dir.")

    parser.add_argument("--save-images", nargs="?", const="output_images",
                        help="Save the card images and photos of detected cards. Optionally provide a path. If not, saves images to 'output_images' in current dir.")

    parser.add_argument('--show', action='store_true',
                        help="Show video.")
    parser.add_argument('--display-card', action='store_true',
                        help="Display one of the cards that appear at a time.")

    parser.add_argument('--deck-list', default=None, type=str,
                        help="Path or link to the deck list.")

    parser.add_argument('--fps', default=60, type=int,
                        help="FPS of the saved video.")

    parser.add_argument('--language', default='EN', type=str,
                        help="Language for card names (e.g. EN, FR, DE, IT, PT). Default: EN.")

    parser.add_argument('--vid-stride', default=1, type=int,
                        help="Process only every Nth video frame (faster on CPU). Default: 1.")

    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Print every detected card on every frame, including why unrecognized ones were rejected.")

    parser.add_argument('--quiet', '-q', action='store_true',
                        help="Do not print new-card events and the final summary.")

    parser.add_argument('--card-info', action='store_true',
                        help="Fetch and print card details (type, ATK/DEF, text) from YGOPRODeck "
                             "the first time each card is recognized. Requires internet.")

    parser.add_argument('--log-json', default=None, type=str,
                        help="Write one JSON line per frame with all detections to this file.")

    return parser.parse_args()


def show(im, p="draw2"):
    cv2.imshow(p, im)
    cv2.waitKey(1)  # 1 millisecond


def save(is_image, outputs, video_writer, save_path):
    if is_image:
        cv2.imwrite(f"{save_path}.png", outputs['image'])
    else:
        video_writer.write(outputs['image'])


def detect_card(outputs, counts, displayed, last_detected, dataset, label2id,
                on_detected: Callable[[OpenCVImage, OpenCVImage, str], None]):
    for label in outputs['predictions']:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1

    for label, count in counts.items():
        if count > 60:
            if label not in displayed:
                displayed[label] = 6
                image = dataset[int(label2id[label])]["image"]
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                if last_detected != label:
                    last_detected = label
                    on_detected(image, outputs['image'], label)
            else:
                displayed[label] -= 1
                if displayed[label] == 0:
                    del displayed[label]
                    counts[label] = 0


def save_images(directory: Path, predicted_image, photo_image, label):
    if directory.is_file():
        raise Exception('save_images <directory> is required to not be a file')

    directory.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    photo_path = str(directory / f'photo_{label}_{timestamp}.jpg')
    image_path = str(directory / f'card_{label}_{timestamp}.jpg')
    cv2.imwrite(photo_path, photo_image)
    cv2.imwrite(image_path, predicted_image)


def get_cache_dir():
    return Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "draw2"


def clear_cache():
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def parse_deck_name(console_entry):
    message = console_entry.get('message', '')
    if not message or '\\"action\\":\\"Success\\"' not in message:
        return ''
    pattern = '\\"name\\":\\"'
    start = message.find(pattern) + len(pattern)
    message = message[start:]
    pattern = '\\",'
    end = message.find(pattern)
    message = message[:end]
    return message


def parse_deck_list(message, dl):
    pattern = '\\"serial_number\\":\\"'
    if pattern not in message:
        return dl
    start = message.find(pattern) + len(pattern)
    message = message[start:]
    pattern = '\\",'
    end = message.find(pattern)
    serial_number = message[:end]
    dl.append(serial_number)

    message = message[end:]
    return parse_deck_list(message, dl)


def get_deck_list(deck_list):
    if os.path.isfile(deck_list):
        return deck_list
    elif deck_list.isdigit() and len(deck_list) == 8:
        options = Options()
        options.add_argument('--headless')
        options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

        url = f"https://www.duelingbook.com/deck?id={deck_list}"

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(3)
        driver.get(url)
        time.sleep(1)

        for entry in driver.get_log('browser'):
            deck_name = parse_deck_name(console_entry=entry) or 'No name'
            if deck_name == 'No name':
                continue

            cache_dir = get_cache_dir()
            cache_dir.mkdir(parents=True, exist_ok=True)
            local_path = cache_dir / f"{deck_name}.ydk"

            db_list = parse_deck_list(entry.get("message"), [])
            db_list = list(set(db_list))

            with open(local_path, 'w') as f:
                for line in db_list:
                    f.write(f"{line}\n")
            return local_path

        return None
    else:
        cache_dir = get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        local_path = cache_dir / f"{deck_list.split('/')[-1]}.ydk"

        if not local_path.exists():
            print(f"Downloading {deck_list} to {local_path}")
            urllib.request.urlretrieve(deck_list, local_path)
        else:
            print(f"Using cached: {local_path}")

        return local_path


def main(args):
    deck_list = None
    if args.deck_list:
        deck_list = get_deck_list(args.deck_list)

    draw = Draw(
        source=args.source,
        deck_lists=[deck_list] if deck_list else None,
        vid_stride=args.vid_stride
    )

    image_types = ["bmp", "dng", "HEIC", "jpeg", "jpg", "mpo", "pfm", "png", "tif", "tiff", "webp"]
    is_image = args.source is not None and args.source.split('.')[-1] in image_types
    if args.save and is_image:
        video_writer = False
        save_path = ".".join(args.save.split('.')[:-1]) if args.save.split(".")[-1] in image_types else args.save

    if args.save and not is_image:
        video_types = ["asf", "avi", "gif", "m4v", "mov", "mp4", "mpeg", "mpg", "ts", "wmv", "webm"]
        first_frame = next(iter(draw.results)).orig_img
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        save_path = ".".join(args.save.split('.')[:-1]) if args.save.split(".")[-1] in video_types else "output"

        video_writer = cv2.VideoWriter(f'{save_path}.avi', fourcc, args.fps,
                                       (first_frame.shape[1], first_frame.shape[0]))

    requires_detection_loop = bool(args.display_card or args.save_images)
    if requires_detection_loop:
        # TODO put those as arguments inside draw class
        counts = {}
        displayed = {}
        last_detected = None

        labels = draw.dataset.features["label"].names
        label2id = dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)

    logger = DetectionLogger(
        verbose=args.verbose,
        card_info=args.card_info,
        json_path=args.log_json,
        vid_stride=args.vid_stride,
        stream=open(os.devnull, 'w') if args.quiet else sys.stdout,
    )

    try:
        frame_start = time.time()
        for frame_idx, result in enumerate(draw.results):
            outputs = draw.process(result, show=bool(args.show or args.save),
                                   display=bool(args.display_card or args.save_images),
                                   language=args.language)

            logger.log_frame(frame_idx, outputs, elapsed=time.time() - frame_start)
            frame_start = time.time()

            if args.show:
                show(outputs['image'])
                # result.show()

            if args.save:
                save(is_image, outputs, video_writer, save_path)

            def on_detected(predicted_image, photo_image, label):
                if args.display_card:
                    show(predicted_image, p='draw2 - Card')
                if args.save_images:
                    save_images(Path(args.save_images).absolute(), predicted_image, photo_image, label)

            if requires_detection_loop:
                detect_card(
                    outputs=outputs,
                    counts=counts,
                    displayed=displayed,
                    last_detected=last_detected,
                    dataset=draw.dataset,
                    label2id=label2id,
                    on_detected=on_detected
                )

        logger.summary()
        if args.show and is_image:
            print("Press any key in the image window to close it.")
            cv2.waitKey(0)
        cv2.destroyAllWindows()
        if args.save and not is_image:
            video_writer.release()
        clear_cache()
    except KeyboardInterrupt:
        logger.summary()
        cv2.destroyAllWindows()
        if args.save and not is_image:
            video_writer.release()
        clear_cache()
    finally:
        logger.close()


if __name__ == '__main__':
    args = parse_command_line()
    main(args)
