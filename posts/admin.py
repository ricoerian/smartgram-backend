from django.contrib import admin
from .models import Post

@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ('user', 'caption', 'status', 'created_at')
    list_filter = ('status', 'created_at')
    search_fields = ('caption',)