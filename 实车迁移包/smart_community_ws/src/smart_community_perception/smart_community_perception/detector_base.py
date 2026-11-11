from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: float = 1.0
    extra: dict = field(default_factory=dict)


class Detector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        ...
