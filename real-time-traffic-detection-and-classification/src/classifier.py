"""
Real-Time Street Traffic Detection and Classification
CS5330 Pattern Recognition & Computer Vision
Drew Hill & Abhiram Banda

classifier.py

"""
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights


TARGET_BY_INDEX = {
    407: "ambulance",
    555: "fire truck",
    569: "garbage truck",
    734: "police car",
    656: "van",
    675: "van",
    717: "pickup truck",
    665: "electric scooter",
    670: "electric scooter",
    609: "suv",
    436: "suv",
    757: "suv",
    468: "car",
    511: "car",
    627: "car",
    817: "car",
    654: "bus",
    779: "bus",
    874: "bus",
    864: "truck",
    866: "truck",
    867: "truck",
}


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
            if index in TARGET_BY_INDEX and score >= self.conf:
                return TARGET_BY_INDEX[index], float(score)
        return None, 0.0
