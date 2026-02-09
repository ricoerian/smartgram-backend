from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0008_alter_post_ai_strength'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='detected_strength_value',
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='post',
            name='strength_auto_detected',
            field=models.BooleanField(default=False),
        ),
        migrations.AlterField(
            model_name='post',
            name='ai_strength',
            field=models.FloatField(blank=True, default=None, help_text='Leave empty for auto-detection based on image quality. Range: 0.20-0.70', null=True),
        ),
    ]
