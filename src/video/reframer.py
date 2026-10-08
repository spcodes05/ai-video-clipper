from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.models.schemas import VideoMetadata
from src.utils.errors import ConfigError


@dataclass(frozen=True)
class CropWindow:
    x: int
    y: int
    width: int
    height: int

    def to_filter(self) -> str:
        return f"crop={self.width}:{self.height}:{self.x}:{self.y}"


def compute_center_crop(src_w: int, src_h: int, target_w: int, target_h: int) -> CropWindow:
    target_ratio = target_w / target_h
    if src_w / src_h > target_ratio:
        h = src_h
        w = int(src_h * target_ratio) // 2 * 2
    else:
        w = src_w
        h = int(src_w / target_ratio) // 2 * 2
    x = (src_w - w) // 2
    y = (src_h - h) // 2
    return CropWindow(x=x, y=y, width=w, height=h)


class Reframer(ABC):
    """Phase 2 plugs face/speaker tracking in here (e.g. return a dynamic crop expression)."""

    def __init__(self, target_w: int = 1080, target_h: int = 1920):
        self.target_w = target_w
        self.target_h = target_h

    @abstractmethod
    def crop_filter(self, meta: VideoMetadata, start: float, end: float) -> str: ...


class CenterCropReframer(Reframer):
    def crop_filter(self, meta: VideoMetadata, start: float, end: float) -> str:
        return compute_center_crop(meta.width, meta.height, self.target_w, self.target_h).to_filter()


def get_reframer(mode: str, target_w: int = 1080, target_h: int = 1920) -> Reframer:
    if mode == "center":
        return CenterCropReframer(target_w, target_h)
    raise ConfigError(f"Unknown reframe_mode '{mode}'. Available: center")