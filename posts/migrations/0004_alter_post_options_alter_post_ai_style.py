from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0003_alter_post_ai_prompt_alter_post_ai_style_and_more'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='post',
            options={'ordering': ['-created_at']},
        ),
        migrations.AlterField(
            model_name='post',
            name='ai_style',
            field=models.CharField(blank=True, choices=[('auto', 'Auto Enhance'), ('noir', 'Film Noir'), ('sepia', 'Vintage Sepia'), ('sketch', 'Pencil Sketch'), ('cyber', 'Cyberpunk'), ('hdr', 'HDR'), ('cartoon', 'Disney Pixar'), ('anime', 'Anime'), ('ghibli', 'Studio Ghibli'), ('realistic', 'Hyperrealistic'), ('oil_painting', 'Oil Painting'), ('watercolor', 'Watercolor'), ('pop_art', 'Pop Art'), ('fantasy', 'Fantasy'), ('steampunk', 'Steampunk'), ('minimalist', 'Minimalist')], default='auto', max_length=20),
        ),
    ]
