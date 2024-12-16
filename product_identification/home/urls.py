from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('video_feed/', views.video_feed, name='video_feed'),
    path('', views.home, name='home'),
    path('stop_video_feed', views.stop_video_feed, name='stop_video_feed'),
    path('captured_images/', views.captured_images, name='captured_images'),
    path('delete_image/<int:image_id>/', views.delete_image, name='delete_image'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)