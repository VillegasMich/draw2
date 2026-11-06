# DRAW 2 — Usage Guide

How to run the card detector from the command line on an **image**, a **video**, or a **live camera**,
with or without a graphical window.

---

## 1. Setup

```shell
# from the repository root, with your virtual environment active
python -m pip install -e .
```

`-e` (editable) means changes in `src/` take effect without reinstalling.

**First run** downloads the models and the card reference dataset from Hugging Face
(~2.5 GB, cached in `~/.cache/huggingface`). Later runs start in a few seconds.

**Performance:** with an NVIDIA GPU (CUDA) processing is near real time. On CPU expect roughly
**0.5–1 s per frame**, so long videos take a while — see `--vid-stride` below.

---

## 2. Quick reference

```shell
python -m draw --source <SOURCE> [options]
```

| Option | Description |
|---|---|
| `--source` | Image path, video path, camera index (`0`, `1`, …) or stream URL. Default: `0` (webcam) |
| `--show` | Open a window with the annotated output |
| `--save [PATH]` | Save the annotated output (image → `.png`, video/camera → `.avi`) |
| `--save-images [DIR]` | Save a photo + the official card image each time a card is confirmed |
| `--display-card` | Open a second window with the official image of the confirmed card |
| `--deck-list` | `.ydk` file to restrict recognition to the cards of a deck |
| `--language` | Card name language: `EN` (default), `FR`, `DE`, `IT`, `PT` |
| `--vid-stride N` | Process only every Nth frame (faster on CPU) |
| `--fps` | FPS of the saved video. Default: 60 |
| `-v`, `--verbose` | Print every detected card on every frame, with the rejection reason when not recognized |
| `--card-info` | Print card details (type, ATK/DEF, text) from YGOPRODeck when a card is first recognized. Needs internet |
| `--log-json FILE` | Write one JSON line per processed frame with all detections |
| `-q`, `--quiet` | Disable the default terminal output |

---

## 3. Terminal output (no window needed)

You don't need a graphical interface to know what was detected. By default the CLI prints:

- a `NEW CARD` line the **first time** each card is recognized
- a **summary** at the end (or when you press `Ctrl+C`)

```
NEW CARD frame 0: Monster Reborn [83764719] confidence 86.1%
NEW CARD frame 280: 7 Completed [86198326] confidence 32.0%

=== Summary ===
Frames processed: 32 in 51.6s (0.62 frames/s)
Card boxes detected: 295
     218  orientation could not be determined
      43  confidence below threshold
      34  recognized
Unique cards recognized: 2
  Monster Reborn [83764719]  in 28 frame(s), best 89.7%, first seen frame 0
  7 Completed [86198326]  in 6 frame(s), best 36.0%, first seen frame 280
```

With `-v` you get every card box on every frame:

```
[frame 0] 9 cards detected, 1 recognized (1.72s)
  ✘ card at (954,678): confidence below threshold  (best guess: Twin-Barrel Dragon 3.4%)
  ✘ card at (957,241): orientation could not be determined
  ✔ Monster Reborn [83764719] 90.8% at (1369,674)  alt: Oscillo Hero 0.0%, Suijin 0.0%
```

With `--card-info`, each new card also shows its details:

```
NEW CARD frame 0: Monster Reborn [83764719] confidence 90.8%
    Normal Spell | Normal
    Target 1 monster in either GY; Special Summon it.
```

Possible statuses of a detected card:

| Status | Meaning |
|---|---|
| `recognized` | Card identified with confidence ≥ 5% (and in the deck list, if one is given) |
| `no_text_box` | No text box found inside the card (blurry, face-down, or not a card) |
| `orientation_unknown` | Could not work out which way up the card is, so it was skipped |
| `low_confidence` | Best guess below the 5% confidence threshold |
| `not_in_deck` | None of the top 15 guesses is in the deck list |

---

## 4. Image

```shell
# print results in the terminal only
python -m draw --source path/to/photo.jpg -v

# open a window (stays open until you press a key in it)
python -m draw --source path/to/photo.jpg --show

# save the annotated image -> result.png
python -m draw --source path/to/photo.jpg --save result.jpg

# all card details + JSON
python -m draw --source path/to/photo.jpg -v --card-info --log-json result.jsonl
```

Try it with the bundled sample: `docs/samples/sample_photo.jpg`.

Supported formats: `bmp`, `dng`, `HEIC`, `jpeg`, `jpg`, `mpo`, `pfm`, `png`, `tif`, `tiff`, `webp`.

---

## 5. Video file

```shell
# terminal only: new cards + summary
python -m draw --source path/to/duel.mp4

# watch it in a window, every 5th frame (5x faster on CPU)
python -m draw --source path/to/duel.mp4 --show --vid-stride 5

# save the annotated video -> duel_annotated.avi
python -m draw --source path/to/duel.mp4 --save duel_annotated.mp4

# full log for later analysis
python -m draw --source path/to/duel.mp4 --vid-stride 10 --log-json duel.jsonl
```

Try it with the bundled sample: `docs/samples/sample_video.mp4` (10 s, 636 frames).

The program exits by itself when the video ends. Press `Ctrl+C` to stop early; the summary is still printed.

> The `--show` window refreshes only once per processed frame. On CPU it looks like slow
> motion and GNOME may briefly mark it "not responding". That is expected.

---

## 6. Live camera (webcam, USB camera, phone)

With a camera as source, the program **keeps running** and keeps analysing the newest frame,
announcing each new card that appears, until you press `Ctrl+C`.
On a slow machine older frames are dropped automatically, so it never falls behind the live feed.

### Built-in webcam

```shell
# headless: just watch the terminal and place cards in front of the camera
python -m draw --source 0

# with a window and the official card image when a card is confirmed
python -m draw --source 0 --show --display-card

# headless, with details of each new card and a full JSON log
python -m draw --source 0 --card-info --log-json session.jsonl
```

### Another camera (USB / external)

Cameras are numbered by index. On Linux, each physical camera often creates **two** device
nodes (`/dev/video0` + `/dev/video1`, metadata), so a second camera is usually index `2`.

```shell
ls /dev/video*                          # list devices
python -m draw --source 2 --show        # second camera
```

### IP camera / phone as camera (not tested yet)

Any RTSP or HTTP video stream URL should work as source. For example, with an Android
"IP Webcam" style app:

```shell
python -m draw --source "http://192.168.1.50:8080/video"
python -m draw --source "rtsp://user:password@192.168.1.60:554/stream1"
```

### Saving a camera session

```shell
python -m draw --source 0 --save session            # -> session.avi
python -m draw --source 0 --save-images captures    # -> captures/photo_<card>_<time>.jpg + card_<card>_<time>.jpg
```

### Notes for live use

- `NEW CARD` is printed **once per card per session**. If you remove a card and put it back,
  it is not announced again. Use `-v` to see what's on the table in every frame.
- `--display-card` and `--save-images` trigger only after a card is recognized in **more than
  60 frames**. On CPU that can take a minute or more.
- Good lighting, a camera pointing down at the mat, and cards facing the camera upright
  give the best results (`orientation_unknown` is the most common failure).

---

## 7. Deck list

Restricting recognition to your deck improves accuracy a lot:

```shell
python -m draw --source 0 --deck-list my_deck.ydk
```

`.ydk` is the standard deck format exported by YGOPRODeck, EDOPro, Dueling Book, etc.
(one card passcode per line).

---

## 8. JSON log format

`--log-json FILE` writes one JSON object per processed frame:

```json
{
  "frame": 0,
  "time": 1793936876.733,
  "processing_seconds": 1.721,
  "detections": [
    {
      "box": [[1422, 749], [1422, 600], [1317, 600], [1316, 748]],
      "center": [1369, 674],
      "detector_confidence": 0.9352,
      "status": "recognized",
      "rotation": 0,
      "candidates": [{"label": "Monster-Reborn-83764719", "card_id": "83764719", "name": "Monster Reborn", "score": 0.9082}],
      "label": "Monster-Reborn-83764719",
      "card_id": "83764719",
      "name": "Monster Reborn",
      "score": 0.9082
    }
  ]
}
```

Example: list every recognized card name with `jq`:

```shell
jq -r '.detections[] | select(.status=="recognized") | .name' session.jsonl | sort | uniq -c
```

---

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `--show` window doesn't appear on Wayland | Run with `QT_QPA_PLATFORM=xcb python -m draw ...` |
| `QFontDatabase: Cannot find font directory` warnings | Harmless, from OpenCV's bundled Qt |
| `ConnectionError: Failed to read images from 0` | Camera can't stream. Check that no other app (browser, Zoom, OBS) is using it, the privacy switch / camera kill key (F-key with a camera icon) is off, and it works in another app such as GNOME "Camera" / `cheese`. Try another index (`--source 1`, `2`) |
| Image window flashes and closes | Update to this version: image windows now wait for a key |
| Very slow | No GPU: use `--vid-stride` for videos, and don't use `--show` if you don't need it |
| `You are sending unauthenticated requests to the HF Hub` | Harmless. Optionally set `HF_TOKEN` |
