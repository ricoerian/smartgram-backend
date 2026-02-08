from django.contrib import admin
from .models import Post


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ('user', 'caption', 'ai_style', 'status', 'created_at')
    list_filter = ('status', 'ai_style', 'use_ai', 'created_at')
    search_fields = ('caption', 'ai_prompt')
    readonly_fields = ('created_at',)