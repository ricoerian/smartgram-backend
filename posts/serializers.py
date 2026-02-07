from rest_framework import serializers
from .models import Post

class PostSerializer(serializers.ModelSerializer):
    username = serializers.ReadOnlyField(source='user.username')

    class Meta:
        model = Post
        fields = [
            'id', 'username', 'caption', 
            'image', 'video', 
            'use_ai', 'ai_style', 'ai_prompt', 
            'status', 'created_at'
        ]