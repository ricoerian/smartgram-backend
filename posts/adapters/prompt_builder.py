from typing import List

from ..config import STYLE_PROMPTS, QUALITY_ENHANCERS
from ..domain.entities import ImageAnalysis


def build_enhanced_prompt(user_prompt: str, style: str, image_analysis: ImageAnalysis) -> str:
    style_text = STYLE_PROMPTS.get(style, STYLE_PROMPTS["auto"])
    
    base_prompt = user_prompt.strip() if user_prompt else "high quality image"
    
    context_hints = []
    
    if image_analysis.is_complex:
        context_hints.append("intricate fine details")
    if image_analysis.has_fine_details:
        context_hints.append("sharp precise textures")
    if image_analysis.is_dark:
        context_hints.append("dramatic moody lighting")
    if image_analysis.is_colorful:
        context_hints.append("rich vibrant colors")
    if image_analysis.is_high_entropy:
        context_hints.append("complex composition")
    
    context_str = ", ".join(context_hints) if context_hints else ""
    
    if context_str:
        raw_text = f"{base_prompt}, {style_text}, {context_str}, {QUALITY_ENHANCERS}"
    else:
        raw_text = f"{base_prompt}, {style_text}, {QUALITY_ENHANCERS}"
    
    tokens = [t.strip() for t in raw_text.split(',') if t.strip()]
    
    seen = set()
    unique = []
    for t in tokens:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique.append(t)
    
    return ", ".join(unique)
