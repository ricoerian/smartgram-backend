from dataclasses import dataclass


@dataclass
class ImageAnalysis:
    edge_density: float
    contrast: float
    brightness: float
    laplacian_variance: float
    color_variance: float
    detail_level: float
    texture_complexity: float
    entropy: float
    is_complex: bool
    is_low_contrast: bool
    is_dark: bool
    is_bright: bool
    has_fine_details: bool
    is_colorful: bool
    is_high_entropy: bool


@dataclass
class ProcessingConfig:
    style: str
    inference_steps: int
    cfg_scale: float
    denoise_strength: float
    use_hires: bool
    use_openpose: bool
    strength_canny: float
    strength_openpose: float
