import cv2
import numpy as np
from PIL import Image, ImageEnhance
from typing import Optional

from ..domain.entities import ImageAnalysis
from ..domain.value_objects import CannyThresholds, ImageDimensions
from ..config import MAX_IMAGE_SIZE
OPENPOSE_AVAILABLE = True
# try:
#     from controlnet_aux import OpenposeDetector
#     import mediapipe
#     OPENPOSE_AVAILABLE = True
# except (ImportError, AttributeError, UserWarning):
#     OPENPOSE_AVAILABLE = False


def analyze_image_complexity(image: Image.Image) -> ImageAnalysis:
    img_array = np.array(image.convert("L"))
    
    edges = cv2.Canny(img_array, 30, 120)
    edge_density = np.sum(edges > 0) / edges.size
    
    contrast = np.std(img_array)
    brightness = np.mean(img_array)
    
    laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
    
    img_rgb = np.array(image)
    color_variance = np.mean([np.std(img_rgb[:,:,i]) for i in range(3)])
    
    blur = cv2.GaussianBlur(img_array, (9, 9), 0)
    detail_level = np.mean(np.abs(img_array.astype(float) - blur.astype(float)))
    
    sobelx = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    texture_complexity = np.mean(gradient_magnitude)
    
    hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    
    return ImageAnalysis(
        edge_density=edge_density,
        contrast=contrast,
        brightness=brightness,
        laplacian_variance=laplacian_var,
        color_variance=color_variance,
        detail_level=detail_level,
        texture_complexity=texture_complexity,
        entropy=entropy,
        is_complex=edge_density > 0.08 or texture_complexity > 25,
        is_low_contrast=contrast < 35,
        is_dark=brightness < 75,
        is_bright=brightness > 185,
        has_fine_details=detail_level > 12,
        is_colorful=color_variance > 45,
        is_high_entropy=entropy > 6.5
    )


def compute_adaptive_canny_thresholds(image: Image.Image) -> CannyThresholds:
    gray = np.array(image.convert("L"))
    
    median_val = np.median(gray)
    std_val = np.std(gray)
    
    sigma = 0.33
    low_threshold = int(max(0, (1.0 - sigma) * median_val))
    high_threshold = int(min(255, (1.0 + sigma) * median_val))
    
    if std_val < 30:
        low_threshold = int(low_threshold * 0.8)
        high_threshold = int(high_threshold * 0.8)
    elif std_val > 70:
        low_threshold = int(low_threshold * 1.2)
        high_threshold = int(high_threshold * 1.2)
    
    low_threshold = max(40, min(low_threshold, 120))
    high_threshold = max(80, min(high_threshold, 220))
    
    return CannyThresholds(low=low_threshold, high=high_threshold)


def preprocess_image(image: Image.Image, analysis: ImageAnalysis) -> Image.Image:
    processed = image.copy()
    
    if analysis.is_low_contrast:
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(1.4)
    
    if analysis.is_dark:
        enhancer = ImageEnhance.Brightness(processed)
        processed = enhancer.enhance(1.25)
    elif analysis.is_bright:
        enhancer = ImageEnhance.Brightness(processed)
        processed = enhancer.enhance(0.88)
    
    if not analysis.has_fine_details:
        enhancer = ImageEnhance.Sharpness(processed)
        processed = enhancer.enhance(1.15)
    
    if analysis.is_colorful:
        enhancer = ImageEnhance.Color(processed)
        processed = enhancer.enhance(1.05)
    
    return processed


def make_canny_condition(image: Image.Image, thresholds: CannyThresholds) -> Image.Image:
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    edges = cv2.Canny(blurred, thresholds.low, thresholds.high)
    
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    
    return Image.fromarray(edges)


def make_openpose_condition(image: Image.Image) -> Optional[Image.Image]:
    if not OPENPOSE_AVAILABLE:
        return None
    openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    result = openpose_detector(image)
    del openpose_detector
    import gc
    gc.collect()
    return result


def smart_resize(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    dimensions = ImageDimensions(width=image.size[0], height=image.size[1])
    
    if dimensions.width > dimensions.height:
        new_width = max_size
        new_height = int(max_size / dimensions.aspect_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * dimensions.aspect_ratio)
    
    new_dims = ImageDimensions(width=new_width, height=new_height)
    new_dims = new_dims.align_to_multiple(64)
    new_dims = new_dims.clamp(256, 1024)
    
    return image.resize(new_dims.to_tuple(), Image.LANCZOS)
