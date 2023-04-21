from django.conf.urls import url
from AnalyticHierarchyProcess import views

urlpatterns = [
    url(r'^api/calc$', views.CalculateAhp),
]
