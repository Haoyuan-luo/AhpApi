# Generated by Django 2.2 on 2023-04-21 07:19

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('AnalyticHierarchyProcess', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ahpmodel',
            name='guidelineMatrix',
            field=models.CharField(default='', max_length=255),
        ),
        migrations.AlterField(
            model_name='ahpmodel',
            name='targetMatrix',
            field=models.CharField(default='', max_length=255),
        ),
    ]
