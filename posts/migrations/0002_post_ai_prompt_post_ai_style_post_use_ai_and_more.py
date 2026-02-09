from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='post',
            name='ai_prompt',
            field=models.TextField(blank=True, help_text='Instruksi spesifik user untuk AI', null=True),
        ),
        migrations.AddField(
            model_name='post',
            name='ai_style',
            field=models.CharField(blank=True, choices=[('auto', 'Auto Enhance (Pencerah)'), ('noir', 'Hitam Putih Sinematik'), ('cartoon', 'Efek Kartun'), ('cyber', 'Cyberpunk (High Contrast)')], default='auto', max_length=20),
        ),
        migrations.AddField(
            model_name='post',
            name='use_ai',
            field=models.BooleanField(default=False, help_text='Apakah user ingin menggunakan AI enhancement?'),
        ),
        migrations.AlterField(
            model_name='post',
            name='status',
            field=models.CharField(choices=[('pending', 'Pending...'), ('processing', 'AI & Compression Processing...'), ('completed', 'Completed!'), ('failed', 'Failed!')], default='pending', max_length=20),
        ),
    ]
