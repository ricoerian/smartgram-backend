from celery import shared_task
import traceback
from .models import Post
from .services import ImageGeneratorService, VideoGeneratorService
from .infrastructure import cleanup_gpu_memory


@shared_task
def process_media_task(post_id: int) -> None:
    try:
        post = Post.objects.get(id=post_id)
        post.status = Post.PostStatus.PROCESSING
        post.save()

        ai_strength = post.ai_strength

        if post.image and post.use_ai:
            base_prompt = post.ai_prompt or ""
            style = post.ai_style or "auto"
            
            result = ImageGeneratorService.generate(
                post.image.path,
                base_prompt,
                style=style,
                strength=ai_strength
            )
            
            if result.get("success"):
                if result.get("was_auto_detected"):
                    post.strength_auto_detected = True
                    post.detected_strength_value = result.get("strength")
                    print(f"✅ Auto-detected strength: {result.get('strength'):.2f}")
                else:
                    post.strength_auto_detected = False
                    print(f"📌 Manual strength used: {ai_strength:.2f}")
            else:
                # Handle generation failure if needed, though service logs it.
                # If result['success'] is False, we might want to fail the task?
                # For now keeping existing logic: if success is False, it just doesn't update strength metadata
                # but let's check if we should raise error to trigger the except block.
                # If the service caught an error, it returns success=False and error message.
                if result.get("error"):
                    raise Exception(result.get("error"))

        if post.video:
            if post.use_ai:
                base_prompt = post.ai_prompt or "high quality video"
                style = post.ai_style or "auto"
                video_strength = ai_strength if ai_strength else 0.30
                # TODO: Handle return value of generate_video
                VideoGeneratorService.generate_video(
                    post.video.path, 
                    base_prompt, 
                    style=style, 
                    strength=video_strength
                )
            else:
                VideoGeneratorService.compress_video(post.video.path)

        post.status = Post.PostStatus.COMPLETED
        post.save()

    except Exception as e:
        print(f"Task error: {e}")
        error_trace = traceback.format_exc()
        try:
            p = Post.objects.get(id=post_id)
            p.status = Post.PostStatus.FAILED
            p.error_message = error_trace
            p.save()
        except Exception:
            pass
    finally:
        cleanup_gpu_memory()