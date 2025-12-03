# prediction/urls.py

from django.urls import path
from . import views

# urlpatterns = [
#     path('', views.get_advice, name='predict_eth_price'),
# ]

urlpatterns = [
    path("", views.home, name="home"),
    path("get_advice/", views.get_advice, name="get_advice"),
]