from celery import shared_task
from .models import Post
from .services import compress_video, generate_ai_image, generate_ai_video, cleanup_gpu_memory


@shared_task
def process_media_task(post_id: int) -> None:
    try:
        post = Post.objects.get(id=post_id)
        post.status = 'processing'
        post.save()

        ai_strength = post.ai_strength

        if post.image and post.use_ai:
            base_prompt = post.ai_prompt or ""
            style = post.ai_style or "auto"
            
            result = generate_ai_image(
                post.image.path,
                base_prompt,
                style=style,
                strength=ai_strength
            )
            
            if isinstance(result, dict) and result.get("success"):
                if result.get("was_auto_detected"):
                    post.strength_auto_detected = True
                    post.detected_strength_value = result.get("strength")
                    print(f"✅ Auto-detected strength: {result.get('strength'):.2f}")
                else:
                    post.strength_auto_detected = False
                    print(f"📌 Manual strength used: {ai_strength:.2f}")

        if post.video:
            if post.use_ai:
                base_prompt = post.ai_prompt or "high quality video"
                style = post.ai_style or "auto"
                video_strength = ai_strength if ai_strength else 0.30
                generate_ai_video(post.video.path, base_prompt, style=style, strength=video_strength)
            else:
                compress_video(post.video.path)

        post.status = 'completed'
        post.save()

    except Exception as e:
        print(f"Task error: {e}")
        try:
            p = Post.objects.get(id=post_id)
            p.status = 'failed'
            p.save()
        except Exception:
            pass
    finally:
        cleanup_gpu_memory()