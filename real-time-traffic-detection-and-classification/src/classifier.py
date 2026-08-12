"""
    Real-Time Street Traffic Detection and Classification
    CS5330 Pattern Recognition & Computer Vision
    Drew Hill & Abhiram Banda

    classifier.py

    This file is used to classify each object using ResNet50. It refines the labels 
    from YOLO detection. We grouped several classes into broader classes for example 
    we groupped mini bus, school bus, and trolly bus into a just the label bus.

    You can see all of out classes we used as well as links ImageNet for each class here:

    https://docs.google.com/spreadsheets/d/1y90yB4PmjNMArtg1sJday6IF0xj7YE3uMmsYEurOAVk/edit?usp=sharing
"""
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
import configs

class Classifier:
    """
        Wrapper class for the ResNet50 model that is used to classify the 
        detected objects

    """
    def __init__(self, conf=0.10, top_k=5):
        """
            Loads the model and sets it up to classify the objects detected.

            conf: float -  Minimum score needed to accept classification.
            top_k: int - How many of the predictions to look at.
        """
        self.conf = conf
        self.top_k = top_k
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self.model.eval()
        self.preprocess = weights.transforms()
        self.categories = weights.meta["categories"]
        self.unmatched_dir = Path(__file__).resolve().parents[1] / "unmatched"

    def classify(self, crop):
        """
            Runs a cropped frame through the ResNet50 and returns the label.

            crop: cv2: Mat - The cropped frame of the object.

            returns the label as a str and the score as a float or the None and
            0.0 if there is no match.
        """
        # Skips empty crops.
        if crop is None or crop.size == 0:
            return None, 0.0

        # Converts the cropped frame from BGR to RGB.
        rgb = np.ascontiguousarray(crop[:, :, ::-1])

        # Converts to a tensor for model.
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        batch = self.preprocess(tensor).unsqueeze(0)

        # Runs the model and turns the outputs into probabilities.
        with torch.no_grad():
            probs = self.model(batch).softmax(1)[0]

        # Gets the k highest scoring matches.
        scores, indices = torch.topk(probs, self.top_k)

        # Returns the first prediction that is in a desired class and is confident enough.
        for score, index in zip(scores.tolist(), indices.tolist()):
            if index in configs.TARGET_BY_INDEX and score >= self.conf:
                return configs.TARGET_BY_INDEX[index], float(score)

        # Saves unmatched objects for analysis of our program.
        self._save_unmatched(crop, int(indices[0]), float(scores[0]))

        return None, 0.0

    def _save_unmatched(self, crop, index, score):
        """
            Saves the cropped frame of the object that failed to be classified
            so this can be analyzed later to improve the program.

            crop: cv2Mat - The cropped frame of the object.
            index: int - The ImageNet class idex that the model predicted.
            score: float - The score of the top prediction.
        """
        # Makes an unmatched folder if it does not exist yet.
        self.unmatched_dir.mkdir(exist_ok=True)

        # Builds the filename.
        name = self.categories[index].replace(" ", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # Saves the info.
        cv2.imwrite(str(self.unmatched_dir / f"{name}_{score:.2f}_{stamp}.jpg"), crop)