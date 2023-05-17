from rest_framework import serializers
from .models import AhpModel


# 序列化模型为其他格式
#
# class AhpSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = AhpModel
#
#         # 序列化所有的字段
#         fields = '__all__'
#
#         # 序列化部分字段
#         # fields = ('id','song','singer','last_modify_date','created')
#
