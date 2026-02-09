from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0005_post_ai_strength'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='post',
            name='ai_strength',
        ),
    ]
