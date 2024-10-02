
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
        accumulative_heat_map = self.heat_map + mask
        self.heat_map = np.clip(accumulative_heat_map, 0, 1)
        colored_mask = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        result_mask = cv2.addWeighted(self.heat_map, 0.5, colored_mask, 0.5, 0)
        return result_mask

    def reset_heat_map(self):
        """
        Reset bản đồ nhiệt
        """
        self.heat_map = np.zeros((self.height, self.width))
        
    def show_heat_map(self, mask, image):
        """
        Hiển thị bản đồ nhiệt
        """
        output_image = cv2.addWeighted(image, 0.6, mask, 0.5, 0)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
        cv2.imshow("heat_map", self.heat_map)
        cv2.waitKey(1)
    
    def time_update_heat_map(self):
        """
        lưu thời gian cập nhật bản đồ nhiệt
        """
        time = 5 # 5 giây cập nhật 1 lần
        return time
