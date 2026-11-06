import json
import os

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoImageProcessor, pipeline
from ultralytics import YOLO

from draw import utils


def download_file(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as file:
        file.write(response.content)


class Draw:
    def __init__(self, source, deck_lists=None, confidence_threshold=5, vid_stride=1):
        self.decklist = []
        if deck_lists is not None:
            for deck_list in deck_lists:
                with open(deck_list) as f:
                    self.decklist += [line.rstrip() for line in f.readlines()]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config = hf_hub_download(repo_id="HichTala/draw2", filename="draw_config.json")
        yolo_path = hf_hub_download(repo_id="HichTala/draw2", filename="ygo_yolo.pt")
        cardnames_path = hf_hub_download(repo_id="HichTala/draw2", filename="cardnames.json")
        with open(config, "rb") as f:
            self.configs = json.load(f)
        with open(cardnames_path, "r", encoding="utf-8") as f:
            self.cardnames = json.load(f)

        model_regression = YOLO(yolo_path)
        self.results = model_regression.predict(
            source=source,
            show_labels=False,
            save=False,
            device=self.device,
            stream=True,
            verbose=False,
            vid_stride=vid_stride
        )

        image_processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            use_fast=True
        )
        self.classifier = pipeline(
            "image-classification",
            model="HichTala/draw2",
            image_processor=image_processor,
            device_map=self.device
        )

        self.dataset = load_dataset("HichTala/ygoprodeck-dataset", split="train")
        labels = self.dataset.features["label"].names
        self.label2id = dict()
        for i, label in enumerate(labels):
            self.label2id[label] = str(i)

        self.confidence_threshold = confidence_threshold

    def card_name(self, label, language='EN'):
        """Human readable card name for a classifier label like 'Monster-Reborn-83764719'."""
        card_id = label.split('-')[-1]
        return self.cardnames.get(card_id, {}).get(language) or ' '.join(label.split('-')[:-1])

    def process(self, result, show=False, display=False, language='EN'):
        """Run recognition on one YOLO result.

        Returns a dict with:
          - 'detections': one entry per detected card box, with its status
            ('recognized', 'no_text_box', 'orientation_unknown', 'low_confidence', 'not_in_deck')
          - 'predictions': recognized labels (only if display=True)
          - 'image': annotated frame (only if show=True)
        """
        outputs = {'detections': []}

        if display:
            outputs['predictions'] = []
        if show:
            outputs['image'] = result.orig_img.copy()

        # if result.obb.id is not None:
        for nbox, boxe in enumerate(result.obb.xyxyxyxyn):
            boxe = np.float32(
                [[b[0] * result.orig_img.shape[1], b[1] * result.orig_img.shape[0]] for b in boxe.cpu()]
            )
            obb = np.intp(boxe)
            xy1, _, xy2, _ = obb

            detection = {
                'box': obb.tolist(),
                'center': [int(c) for c in boxe.mean(axis=0)],
                'detector_confidence': round(float(result.obb.conf[nbox]), 4),
                'status': 'no_text_box',
            }
            outputs['detections'].append(detection)

            output_pts = np.float32([
                [224, 224],
                [224, 0],
                [0, 0],
                [0, 224]
            ])
            perspective_transform = cv2.getPerspectiveTransform(boxe, output_pts)
            roi = cv2.warpPerspective(
                result.orig_img, perspective_transform, (224, 224), flags=cv2.INTER_LINEAR
            )
            contours = utils.extract_contours(
                roi,
                d=self.configs["bilateral_filter_d"],
                sigma_color=self.configs["bilateral_filter_sigma_color"],
                sigma_space=self.configs["bilateral_filter_sigma_space"],
                thresh=self.configs["txt_box_contour_threshold"]
            )

            if contours != ():

                if show:
                    cv2.drawContours(outputs['image'], [obb], 0, (152, 255, 119), 2)

                contour = contours[np.array(list(map(cv2.contourArea, contours))).argmax()]
                box_txt, txt_aspect_ratio = utils.get_txt(contour)

                rotation = utils.get_rotation(boxes=result.obb.xywhr[nbox], box_txt=box_txt)
                if rotation is None:
                    detection['status'] = 'orientation_unknown'
                    continue

                if rotation != utils.NO_ROTATION:
                    roi = cv2.rotate(roi, rotation)
                detection['rotation'] = {utils.NO_ROTATION: 0, cv2.ROTATE_90_CLOCKWISE: 90, cv2.ROTATE_180: 180,
                                         cv2.ROTATE_90_COUNTERCLOCKWISE: 270}[rotation]

                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = Image.fromarray(roi)

                output = self.classifier(roi, top_k=15)
                detection['candidates'] = [
                    {'label': o['label'], 'card_id': o['label'].split('-')[-1],
                     'name': self.card_name(o['label'], language), 'score': round(float(o['score']), 4)}
                    for o in output[:3]
                ]

                if output[0]['score'] < self.confidence_threshold / 100:
                    detection['status'] = 'low_confidence'
                    continue

                match = None
                if len(self.decklist) == 0:
                    match = output[0]
                else:
                    for candidate in output:
                        if candidate['label'].split('-')[-1] in self.decklist:
                            match = candidate
                            break

                if match is None:
                    detection['status'] = 'not_in_deck'
                    continue

                label = match['label']
                detection.update({
                    'status': 'recognized',
                    'label': label,
                    'card_id': label.split('-')[-1],
                    'name': self.card_name(label, language),
                    'score': round(float(match['score']), 4),
                })
                if display:
                    outputs['predictions'].append(label)
                if show:
                    outputs['image'] = utils.put_text(outputs['image'], detection['name'],
                                                      (xy1[0], xy1[1]))
        return outputs
