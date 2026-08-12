"""
Real-Time Street Traffic Detection and Classification
CS5330 Pattern Recognition & Computer Vision
Drew Hill & Abhiram Banda

classifier.py

"""
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights

import configs


def crop_object(frame, x1, y1, x2, y2, pad=0.10):
    """


    """
    h, w = frame.shape[:2]
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    return frame[max(0, int(y1 - py)):min(h, int(y2 + py)),
                 max(0, int(x1 - px)):min(w, int(x2 + px))]


class Classifier:
    """


    """
    def __init__(self, conf=0.10, top_k=5):
        """


        """
        self.conf = conf
        self.top_k = top_k
        weights = ResNet50_Weights.IMAGENET1K_V2
        self.model = resnet50(weights=weights)
        self.model.eval()
        self.preprocess = weights.transforms()
        self.categories = weights.meta["categories"]
        self.unmatched_dir = Path(__file__).resolve().parents[1] / "unmatched"

    def predict(self, crop):
        """


        """
        if crop is None or crop.size == 0:
            return None, 0.0
        rgb = np.ascontiguousarray(crop[:, :, ::-1])
        tensor = torch.from_numpy(rgb).permute(2, 0, 1)
        batch = self.preprocess(tensor).unsqueeze(0)
        with torch.no_grad():
            probs = self.model(batch).softmax(1)[0]
        scores, indices = torch.topk(probs, self.top_k)
        for score, index in zip(scores.tolist(), indices.tolist()):
            if index in configs.TARGET_BY_INDEX and score >= self.conf:
                return configs.TARGET_BY_INDEX[index], float(score)
        self._save_unmatched(crop, int(indices[0]), float(scores[0]))
        return None, 0.0

    def _save_unmatched(self, crop, index, score):
        """


        """
        self.unmatched_dir.mkdir(exist_ok=True)
        name = self.categories[index].replace(" ", "_")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(str(self.unmatched_dir / f"{name}_{score:.2f}_{stamp}.jpg"), crop)
