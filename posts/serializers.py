from rest_framework import serializers
from .models import Post


class PostSerializer(serializers.ModelSerializer):
    username = serializers.ReadOnlyField(source='user.username')

    class Meta:
        model = Post
        fields = [
            'id',
            'username',
            'caption',
            'image',
            'video',
            'use_ai',
            'ai_style',
            'ai_prompt',
            'ai_strength',
            'status',
            'created_at'
        ]
    
    def validate(self, data):
        image = data.get('image')
        video = data.get('video')
        use_ai = data.get('use_ai', False)
        ai_prompt = data.get('ai_prompt')
        
        if not image and not video:
            if not (use_ai and ai_prompt):
                raise serializers.ValidationError(
                    "Either provide an image/video file, or enable AI generation with a prompt (use_ai=true and ai_prompt)."
                )
        
        return data