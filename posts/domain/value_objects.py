from dataclasses import dataclass
from typing import Tuple
import torch


@dataclass(frozen=True)
class DeviceConfig:
    device: str
    dtype: torch.dtype
    
    @classmethod
    def from_cuda_availability(cls) -> 'DeviceConfig':
        if torch.cuda.is_available():
            return cls(device="cuda", dtype=torch.float16)
        return cls(device="cpu", dtype=torch.float32)


@dataclass(frozen=True)
class ImageDimensions:
    width: int
    height: int
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.width, self.height)
    
    def scale(self, factor: float) -> 'ImageDimensions':
        new_width = int(self.width * factor)
        new_height = int(self.height * factor)
        return ImageDimensions(width=new_width, height=new_height)
    
    def align_to_multiple(self, multiple: int = 64) -> 'ImageDimensions':
        aligned_width = (self.width // multiple) * multiple
        aligned_height = (self.height // multiple) * multiple
        return ImageDimensions(width=aligned_width, height=aligned_height)
    
    def clamp(self, min_size: int, max_size: int) -> 'ImageDimensions':
        clamped_width = max(min_size, min(self.width, max_size))
        clamped_height = max(min_size, min(self.height, max_size))
        return ImageDimensions(width=clamped_width, height=clamped_height)


@dataclass(frozen=True)
class StrengthValue:
    value: float
    
    def __post_init__(self):
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(f"Strength must be between 0.0 and 1.0, got {self.value}")
    
    def clamp(self, min_val: float = 0.15, max_val: float = 0.75) -> 'StrengthValue':
        clamped = max(min_val, min(self.value, max_val))
        return StrengthValue(value=clamped)
    
    def __float__(self) -> float:
        return self.value


@dataclass(frozen=True)
class CannyThresholds:
    low: int
    high: int
    
    def __post_init__(self):
        if not 0 <= self.low <= 255:
            raise ValueError(f"Low threshold must be between 0 and 255, got {self.low}")
        if not 0 <= self.high <= 255:
            raise ValueError(f"High threshold must be between 0 and 255, got {self.high}")
        if self.low >= self.high:
            raise ValueError(f"Low threshold must be less than high threshold")
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.low, self.high)
