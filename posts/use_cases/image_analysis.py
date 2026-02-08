from ..domain.entities import ImageAnalysis
from ..domain.value_objects import StrengthValue
from ..config.ai_config import STYLE_STRENGTH_MODIFIERS


def compute_optimal_strength(analysis: ImageAnalysis, style: str) -> StrengthValue:
    base_strength = 0.35
    
    edge_density = analysis.edge_density
    if edge_density > 0.25:
        base_strength -= 0.25
    elif edge_density > 0.15:
        base_strength -= 0.15
    elif edge_density > 0.10:
        base_strength -= 0.08
    elif edge_density < 0.04:
        base_strength += 0.18
    elif edge_density < 0.06:
        base_strength += 0.12
    
    sharpness = analysis.laplacian_variance
    if sharpness > 1000:
        base_strength -= 0.18
    elif sharpness > 600:
        base_strength -= 0.10
    elif sharpness > 400:
        base_strength -= 0.05
    elif sharpness < 80:
        base_strength += 0.20
    elif sharpness < 150:
        base_strength += 0.12
    
    contrast = analysis.contrast
    if contrast > 75:
        base_strength -= 0.10
    elif contrast > 60:
        base_strength -= 0.05
    elif contrast < 25:
        base_strength += 0.18
    elif contrast < 40:
        base_strength += 0.10
    
    detail = analysis.detail_level
    if detail > 25:
        base_strength -= 0.12
    elif detail > 18:
        base_strength -= 0.06
    elif detail < 6:
        base_strength += 0.15
    elif detail < 10:
        base_strength += 0.08
    
    color_var = analysis.color_variance
    if color_var > 70:
        base_strength -= 0.08
    elif color_var > 55:
        base_strength -= 0.04
    elif color_var < 25:
        base_strength += 0.10
    elif color_var < 35:
        base_strength += 0.06
    
    texture = analysis.texture_complexity
    if texture > 35:
        base_strength -= 0.10
    elif texture < 15:
        base_strength += 0.08
    
    entropy = analysis.entropy
    if entropy > 7.0:
        base_strength -= 0.08
    elif entropy < 5.5:
        base_strength += 0.10
    
    if analysis.is_dark:
        base_strength += 0.10
    elif analysis.is_bright:
        base_strength += 0.06
    
    if sharpness > 600 and contrast > 55 and edge_density > 0.12:
        base_strength -= 0.12
    
    if sharpness < 120 and contrast < 35 and edge_density < 0.06:
        base_strength += 0.15
    
    base_strength += STYLE_STRENGTH_MODIFIERS.get(style, 0.0)
    
    strength = StrengthValue(value=base_strength).clamp(0.15, 0.75)
    
    print(f"Intelligent strength computed: {float(strength):.3f} (edge: {edge_density:.3f}, sharpness: {sharpness:.1f}, contrast: {contrast:.1f})")
    return strength
