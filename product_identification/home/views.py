from django.shortcuts import render, redirect, get_object_or_404

# Create your views here.
from django.http import StreamingHttpResponse, JsonResponse
import cv2
from ultralytics import YOLO
import os
from .models import CapturedImage
from django.core.files.base import ContentFile
import io
import time
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# Load the YOLO model
model = YOLO("./../../train/weights/best.pt")

# Global variable to control video streaming
stop_streaming = False

def home(request):
    return render(request, 'home.html')


def gen_frames():
    global stop_streaming
    cap = cv2.VideoCapture(0)
    global current_frame
    channel_layer = get_channel_layer()
    
    last_capture_time = time.time()  # Thời gian chụp lần cuối
    confidence_threshold = 0.85  # Tăng ngưỡng độ tin cậy

    while True:
        if stop_streaming:
            break

        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Filter results based on confidence threshold
        filtered_boxes = [box for box in results[0].boxes if box.conf[0] > confidence_threshold]

        # Check if there are any results to plot
        if filtered_boxes:
            frame = results[0].plot(show=False, conf=True)

            # Check if any box has confidence >= 0.9 and save the frame
            for box in filtered_boxes:
                if box.conf[0] >= 0.9:
                    current_time = time.time()
                    if current_time - last_capture_time >= 2:  # Kiểm tra thời gian
                        # Save the frame to the database
                        _, buffer = cv2.imencode('.jpg', frame)
                        image_file = ContentFile(buffer.tobytes())
                        captured_image = CapturedImage()
                        captured_image.image.save('captured_frame.jpg', image_file)

                        last_capture_time = current_time  # Cập nhật thời gian chụp lần cuối
                        
                        # Khựng lại 1 nhịp
                        time.sleep(0.5)  # Tạm dừng 0.5 giây (có thể điều chỉnh theo ý muốn)
                    break  # Save only once per frame

        # Store the current frame for capture
        current_frame = frame

        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


def video_feed(request):
    global stop_streaming
    stop_streaming = False  # Reset the stop flag when starting the feed
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def stop_video_feed(request):
    global stop_streaming
    stop_streaming = True
    return JsonResponse({'status': 'stopped'})

def captured_images(request):
    images = CapturedImage.objects.all().order_by('-timestamp')  # Fetch images ordered by latest
    return render(request, 'captured_images.html', {'images': images})

def delete_image(request, image_id):
    image = get_object_or_404(CapturedImage, id=image_id)
    image.delete()  # Xóa ảnh
    return redirect('captured_images')  # Chuyển hướng về trang danh sách ảnh
