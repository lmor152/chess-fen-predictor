from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset

from chessing import N_RANKS, SQUARE_SIZE


def board2squares(board_img):
    squares = []

    for row in range(N_RANKS):
        for col in range(N_RANKS):
            x1, y1 = col * SQUARE_SIZE, row * SQUARE_SIZE
            x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE

            squares.append(board_img[y1:y2, x1:x2])

    return squares


class ChessDataset(Dataset):
    def __init__(self, image_paths: list[str | Path], transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        fen = get_image_fen(img_path)
        labels = fen2labels(fen, encode=True)

        board_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        square_imgs = board2squares(board_img)

        # Apply transformations if needed
        if self.transform:
            square_imgs = [self.transform(img) for img in square_imgs]

        return torch.stack(square_imgs), torch.tensor(labels)
