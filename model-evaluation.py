# model-evaluation.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Đọc dữ liệu từ file CSV
df = pd.read_csv("./train/results.csv")

accuracy = df['metrics/mAP50(B)'].mean()  # Tính trung bình Accuracy (hoặc cột tương tự)

# Tính toán các chỉ số từ các cột có sẵn
precision = df['metrics/precision(B)'].mean()  # Tính trung bình Precision
recall = df['metrics/recall(B)'].mean()        # Tính trung bình Recall
# F1-score có thể được tính từ Precision và Recall
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# In kết quả
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')

# Tính toán Loss Function
train_box_loss = df['train/box_loss']
train_cls_loss = df['train/cls_loss']
train_dfl_loss = df['train/dfl_loss']
val_box_loss = df['val/box_loss']
val_cls_loss = df['val/cls_loss']
val_dfl_loss = df['val/dfl_loss']

# Dữ liệu cho các chỉ số
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
values = [accuracy, precision, recall, f1]  # Thay đổi giá trị này theo kết quả của bạn

# Tạo biểu đồ cột
colors = ['blue', 'orange', 'green', 'red']
plt.bar(labels, values, color=colors)

# Thêm tiêu đề và nhãn
plt.title('Các chỉ số đánh giá mô hình')
plt.ylabel('Giá trị')
plt.ylim(0, 1)  # Giới hạn trục y từ 0 đến 1

# Vẽ biểu đồ Loss Function
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], train_box_loss, label='Train Box Loss')
plt.plot(df['epoch'], train_cls_loss, label='Train Class Loss')
plt.plot(df['epoch'], train_dfl_loss, label='Train DFL Loss')
plt.plot(df['epoch'], val_box_loss, label='Validation Box Loss')
plt.plot(df['epoch'], val_cls_loss, label='Validation Class Loss')
plt.plot(df['epoch'], val_dfl_loss, label='Validation DFL Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Function Over Epochs')
plt.legend()

# Hiển thị biểu đồ
plt.show()

