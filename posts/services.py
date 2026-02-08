import os

try:
    import ffmpeg
    FFMPEG_AVAILABLE = True
except ImportError:
    FFMPEG_AVAILABLE = False

from .use_cases.ai_generation import generate_ai_image
from .infrastructure.memory_manager import cleanup_gpu_memory


def generate_ai_video(video_path: str, prompt: str, style: str = "auto", strength: float = 0.3) -> bool:
    return False


def compress_video(video_path: str) -> bool:
    if not FFMPEG_AVAILABLE:
        return False
    
    try:
        output_path = video_path.replace('.mp4', '_temp.mp4')
        (
            ffmpeg.input(video_path)
            .output(output_path, vcodec='libx264', crf=23, preset='slow', movflags='faststart')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        os.remove(video_path)
        os.rename(output_path, video_path)
        return True
    except Exception:
        return False


__all__ = [
    'generate_ai_image',
    'generate_ai_video',
    'compress_video',
    'cleanup_gpu_memory',
]