import cv2
import numpy as np

def canny_edge_detection(image_path):
    """Mendeteksi tepi menggunakan metode Canny Edge Detection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(img_blur, 50, 150)

    cv2.imshow("Canny Edge Detection", edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("canny_result.jpg", edges)

    return "canny_result.jpg"

canny_edge_detection(r"D:\Tugas Mei\semester 7\Prak. Kontrol Cerdas\week6\rail_segmentation\test\images\1000195092_0014-0_jpeg.rf.bdcc36fa5199304696bb201c6db9c07c.jpg")