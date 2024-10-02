"""
File này sẽ cấu hình bản đồ nhiệt và các hàm liên quan tới cấu hình
"""
import numpy as np
import cv2
class heat_map:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.heat_map = np.zeros((height, width))
        
    def update_heat_map(self, mask):
        """
        Cập nhật bản đồ nhiệt
        Args:
            mask: ma trận mask của người
        """
        self.heat_map += mask
        # chia mean để chuẩn hóa để giữ cho bản đồ nhiệt không bị quá lớn
        self.heat_map = self.heat_map / np.mean(self.heat_map)
        self.heat_map = np.clip(self.heat_map, 0, 1) # giữ cho giá trị nằm trong khoảng 0-1

    def get_heat_map(self):
        """
        Lấy bản đồ nhiệt
        Returns:
            heat_map: bản đồ nhiệt
        """
        return self.heat_map

    def reset_heat_map(self):
        """
        Reset bản đồ nhiệt
        """
        self.heat_map = np.zeros((self.height, self.width))
        
    def get_heat_map_color(self):
        """
        Lấy bản đồ nhiệt màu
        Returns:
            heat_map_color: bản đồ nhiệt màu
        """
        heat_map_color = cv2.applyColorMap(np.uint8(255*self.heat_map), cv2.COLORMAP_JET)
        return heat_map_color
    
    def time_update_heat_map(self):
        """
        lưu thời gian cập nhật bản đồ nhiệt
        """
        time = 5 # 5 giây cập nhật 1 lần
        return time
