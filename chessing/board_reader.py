import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from chessing import LABELS
from chessing.dataset import board2squares
from chessing.model import ChessPieceClassifier

Image = np.ndarray | cv2.typing.MatLike
FEN = str


class BoardReader:
    def __init__(self, model_path: Path) -> None:
        self.device = self.get_device()

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert image to tensor
                transforms.Normalize(
                    mean=[0.5], std=[0.5]
                ),  # Apply normalization, adapt to your training data
            ]
        )

        model = ChessPieceClassifier()
        model.load_state_dict(torch.load(model_path))
        model.to(self.device)
        model.eval()
        self.model = model

    def get_device(self) -> torch.device:
        try:
            if torch.cuda.is_available():
                return torch.device("cuda")
        except Exception:
            pass

        try:
            if torch.mps.is_available():
                return torch.device("mps")
        except Exception:
            pass

        return torch.device("cpu")

    def model_inference(self, board_img: Image) -> FEN:
        if len(board_img.shape) == 3:
            board_img = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

        # resize the image to 400x400
        board_img = cv2.resize(board_img, (400, 400))

        square_images = board2squares(board_img)

        inputs = [self.transform(square) for square in square_images]
        inputs = torch.stack(inputs).unsqueeze(0).to(self.device)  # Add batch dimension

        with torch.no_grad():
            outputs = self.model(inputs)

        _, predicted = torch.max(outputs, 2)
        predicted_labels = predicted.cpu().numpy()
        return labels2fen(predicted_labels[0], encoded=True)

    def fen_from_image_path(self, img_path: Path) -> FEN:
        board_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        return self.model_inference(board_img)

    def fen_from_image(self, img: Image) -> FEN:
        return self.model_inference(img)


def fen2labels(fen: str, encode: bool = False) -> list[str] | list[int]:
    ranks = fen.split("/")

    labels: list[str] = []
    for rank in ranks:
        for square in rank:
            if square.isdigit():
                labels.extend(["0"] * int(square))
            elif square in LABELS:
                labels.append(square)

    if encode:
        return [LABELS.index(label) for label in labels]

    return labels


def labels2fen(labels: list[str], encoded: bool = False) -> str:
    if encoded:
        labels = [LABELS[label] for label in labels]

    fen = "/".join("".join(row) for row in np.reshape(labels, (8, 8)))
    return re.sub(r"0+", lambda m: str(len(m.group(0))), fen)
