# Generated by Django 2.2 on 2023-04-21 04:36

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='AhpModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('guidelineMatrix', models.CharField(default='', max_length=100)),
                ('targetMatrix', models.CharField(default='', max_length=100)),
                ('version', models.IntegerField(default='', max_length=100)),
                ('created', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'calculation_matrix',
            },
        ),
    ]
