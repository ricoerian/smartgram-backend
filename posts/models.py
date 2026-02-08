import os
from django.contrib.auth.models import User
from django.db import models, transaction
from django.db.models.signals import post_delete
from django.dispatch import receiver


class Post(models.Model):
    STYLE_CHOICES = [
        ('auto', 'Auto Enhance'),
        ('noir', 'Film Noir'),
        ('sepia', 'Vintage Sepia'),
        ('sketch', 'Pencil Sketch'),
        ('cyber', 'Cyberpunk'),
        ('hdr', 'HDR'),
        ('cartoon', 'Disney Pixar'),
        ('anime', 'Anime'),
        ('ghibli', 'Studio Ghibli'),
        ('realistic', 'Hyperrealistic'),
        ('oil_painting', 'Oil Painting'),
        ('watercolor', 'Watercolor'),
        ('pop_art', 'Pop Art'),
        ('fantasy', 'Fantasy'),
        ('steampunk', 'Steampunk'),
        ('minimalist', 'Minimalist'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    caption = models.TextField(blank=True)
    image = models.ImageField(upload_to='posts/images/', blank=True, null=True)
    video = models.FileField(upload_to='posts/videos/', blank=True, null=True)
    use_ai = models.BooleanField(default=False)
    ai_style = models.CharField(max_length=20, choices=STYLE_CHOICES, default='auto', blank=True)
    ai_prompt = models.TextField(blank=True, null=True)
    
    ai_strength = models.FloatField(
        default=None, 
        null=True, 
        blank=True,
        help_text="Leave empty for auto-detection based on image quality. Range: 0.20-0.70"
    )
    
    strength_auto_detected = models.BooleanField(default=False)
    detected_strength_value = models.FloatField(null=True, blank=True)
    
    status = models.CharField(max_length=20, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self) -> str:
        return f"{self.user.username} - {self.ai_style}"

    def save(self, *args, **kwargs) -> None:
        is_new = self.pk is None
        super().save(*args, **kwargs)
        if is_new and (self.image or self.video):
            from .tasks import process_media_task
            transaction.on_commit(lambda: process_media_task.delay(self.pk))
    
    def get_strength_display(self) -> str:
        if self.strength_auto_detected and self.detected_strength_value:
            return f"{self.detected_strength_value:.2f} (Auto-detected)"
        elif self.ai_strength:
            return f"{self.ai_strength:.2f} (Manual)"
        else:
            return "Auto (akan dideteksi)"


@receiver(post_delete, sender=Post)
def delete_file_on_remove(sender, instance: Post, **kwargs) -> None:
    if instance.image and os.path.isfile(instance.image.path):
        os.remove(instance.image.path)

    if instance.video and os.path.isfile(instance.video.path):
        os.remove(instance.video.path)