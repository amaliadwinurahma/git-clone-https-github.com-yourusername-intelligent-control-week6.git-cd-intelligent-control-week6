import cv2
import numpy as np
import os

def canny_edge_detection_comparison(image_path):
    """Bandingkan hasil deteksi tepi dengan berbagai parameter Canny"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Beberapa kombinasi threshold
    params = [
        (30, 100),   # lebih sensitif (banyak garis halus)
        (50, 150),   # standar (seimbang)
        (100, 200),  # kurang sensitif (hanya garis kuat)
    ]

    results = []

    for i, (low, high) in enumerate(params):
        edges = cv2.Canny(img_blur, low, high)
        filename = f"canny_{low}_{high}.jpg"
        cv2.imwrite(filename, edges)
        results.append((filename, edges))

    # Tampilkan hasil berdampingan
    combined = np.hstack([r[1] for r in results])
    cv2.imshow("Perbandingan Canny (Low-High: 30-100 | 50-150 | 100-200)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("canny_comparison.jpg", combined)
    print("Hasil disimpan sebagai canny_comparison.jpg")

canny_edge_detection_comparison(
    r"D:\Tugas Mei\semester 7\Prak. Kontrol Cerdas\week6\rail_segmentation\test\images\1000195092_0081-0_jpeg.rf.e1e804f31d164e3431431e37e41f1f8a.jpg"
)
