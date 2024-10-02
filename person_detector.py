"""
    Module này se dùng để nhận diện người trong ảnh
    Các hàm chính:
        - detect_person: nhận diện người trong ảnh
        - get_bounding_box: lấy bounding box của người trong ảnh
        ...
    *** SAU NÀY XÓA PHẦN NÀY ***
    các bước để nhận diện người:
    - dùng model yolov8 để nhận diện người
    - lấy bounding box của người
    - tạo matrix mask của người (chỉ có người là 1, còn lại là 0) để dùng cho bản đồ nhiệt
    - tạo bản đồ nhiệt từ matrix mask
    - trả về matrix bản đồ nhiệt
"""
# Tiến hành
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("best.pt")

def detect_person(image):
    """
    Nhận diện người trong ảnh
    Args:
        image: ảnh đầu vào
    Returns:
        bounding_box: bounding box của người
    """
    results = model(image)
    bounding_box = results.xyxy[0].cpu().numpy()
    return bounding_box # dưới dạng [x1, y1, x2, y2, confidence, class]

def bounding_box_to_ones_matrix(bounding_box, width, height):
    """
    Chuyển bounding box sang ma trận mask
    Args:
        bounding_box: bounding box của người
        width: chiều rộng ảnh
        height: chiều cao ảnh
    Returns:
        mask: ma trận mask của người
    """
    mask = np.zeros((height, width))
    for box in bounding_box:
        x1, y1, x2, y2 = box[:4].astype(int)
        mask[y1:y2, x1:x2] = 1
    return mask

if __name__ == "__main__":
    pass