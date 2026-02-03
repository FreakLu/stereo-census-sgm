from dataclasses import dataclass

@dataclass
class AnnPoint:
    label: str
    x: float
    y: float

@dataclass 
class PointDepthResult:
    label: str
    x_left: float
    y_left: float
    x_right: float
    y_right: float
    disparity: float
    depth: float
    cost: int
    gap: int
