"""JREproject URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from pages.views import home_view
from pages.views import plot_view
from pages.views import captions_view
from pages.views import tfidfGuest_view
from pages.views import tfidfSearch_view
from pages.views import nameSearch_view

urlpatterns = [
    path('', home_view, name='home'),
    path('plotting', plot_view, name='plotting'),
    path('tfidf_guest', tfidfGuest_view, name='tfidfGuest'),
    path('tfidf_search', tfidfSearch_view, name='tfidfSearch'),
    path('name_search', nameSearch_view, name='nameSearch'),
    path('captions', captions_view, name='captions'),
    path('admin/', admin.site.urls),
] 
