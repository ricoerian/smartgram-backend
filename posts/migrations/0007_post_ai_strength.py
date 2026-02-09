from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0006_remove_post_ai_strength'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='ai_strength',
            field=models.FloatField(default=0.35, help_text='0.1 (Mirip Asli) - 0.9 (Imajinasi Liar)'),
        ),
    ]
