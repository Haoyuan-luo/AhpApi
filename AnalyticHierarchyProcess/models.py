from django.db import models


# Create your models here.
class AhpModel(models.Model):
    guidelineMatrix = models.CharField(max_length=255, default='')
    targetMatrix = models.CharField(max_length=255, default='')
    version = models.IntegerField(max_length=100, default='')
    created = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = 'calculation_matrix'
