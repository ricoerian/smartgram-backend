from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('posts', '0007_post_ai_strength'),
    ]

    operations = [
        migrations.AlterField(
            model_name='post',
            name='ai_strength',
            field=models.FloatField(default=0.35),
        ),
    ]
