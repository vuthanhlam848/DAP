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

class person_detector:
    def __init__(self, model):
        self.model = YOLO("best.pt")
        
    def detect_person(self, image):
        """
        hàm nhận diện người trong ảnh
        return:
            list bounding box của người
        """
        results = self.model(image)
        return results.xyxy[0].numpy() # bounding box của người
    
    def bouding_box_to_mask(self, bounding_box, width, height):
        """
        chuyển bounding box sang mask
        """
        mask = np.zeros((height, width))
        for box in bounding_box:
            x1, y1, x2, y2 = box
            mask[int(y1):int(y2), int(x1):int(x2)] = 1 #vùng có người là 1, còn lại là 0
        return mask
    
    def count_person(self, bounding_box):
        """
        đếm số người trong ảnh
        """
        return len(bounding_box)
    
    def __call__(self, image):
        """
        hàm gọi hàm detect_person
        """
        return self.detect_person(image)