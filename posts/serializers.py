from rest_framework import serializers
from .models import Post


class PostSerializer(serializers.ModelSerializer):
    username = serializers.ReadOnlyField(source='user.username')

    class Meta:
        model = Post
        fields = [
            'id', 'username', 'caption', 'image', 'video',
            'use_ai', 'ai_style', 'ai_prompt', 'ai_strength',
            'generation_type', 'video_prompt',  # <-- tambah ini
            'status', 'created_at'
        ]

    def validate(self, data):
        gen_type = data.get('generation_type')
        use_ai = data.get('use_ai', False)
        image = data.get('image')
        video = data.get('video')  # mungkin tidak perlu di awal
        prompt = data.get('ai_prompt')

        if not use_ai:
            if not image and not video:
                raise serializers.ValidationError("Harus upload image atau video jika tidak pakai AI.")
            return data

        if not prompt:
            raise serializers.ValidationError("ai_prompt wajib diisi jika use_ai=True.")

        if gen_type in [GenerationType.IMAGE_TO_IMAGE, GenerationType.IMAGE_TO_VIDEO] and not image:
            raise serializers.ValidationError(f"Harus upload image jika memilih {gen_type.label}.")

        if gen_type in [GenerationType.TEXT_TO_VIDEO, GenerationType.IMAGE_TO_VIDEO] and not data.get('video'):
            # Video field akan diisi nanti oleh task, jadi boleh kosong di awal
            pass

        return data