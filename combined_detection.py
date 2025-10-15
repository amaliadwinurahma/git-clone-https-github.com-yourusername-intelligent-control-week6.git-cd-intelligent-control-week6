import cv2
import numpy as np
import os

def canny_edge_detection(image_path, low_thresh=50, high_thresh=150):
    """Deteksi tepi rel menggunakan metode Canny Edge Detection"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"⚠️ Gambar tidak ditemukan di: {image_path}")

    # ✨ Naikkan kontras agar tepi lebih jelas
    img_eq = cv2.equalizeHist(img)
    img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

    edges = cv2.Canny(img_blur, low_thresh, high_thresh)
    if np.count_nonzero(edges) < 500:
        print("⚠️ Canny terlalu lemah, pakai threshold lebih sensitif...")
        edges = cv2.Canny(img_blur, 20, 80)

    cv2.imwrite("canny_result.jpg", edges)
    print("✅ Hasil Canny Edge disimpan sebagai 'canny_result.jpg'")
    return edges


def region_of_interest(img):
    height, width = img.shape[:2]
    polygons = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.7), int(height * 0.6)),
        (int(width * 0.3), int(height * 0.6))
    ]])
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygons, 255)
    return cv2.bitwise_and(img, mask)


def average_slope_intercept(lines):
    left_fit, right_fit = [], []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        if slope < -0.1:
            left_fit.append((slope, intercept))
        elif slope > 0.1:
            right_fit.append((slope, intercept))
    left_avg = np.average(left_fit, axis=0) if left_fit else None
    right_avg = np.average(right_fit, axis=0) if right_fit else None
    return left_avg, right_avg


def make_coordinates(img, line_params):
    slope, intercept = line_params
    y1 = img.shape[0]
    y2 = int(y1 * 0.55)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def draw_lines(img, lines, color_left=(0, 0, 255), color_right=(0, 255, 0), thickness=10):
    if lines is None:
        return img
    left_line, right_line = lines
    if left_line is not None:
        coords = make_coordinates(img, left_line)
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), color_left, thickness)
    if right_line is not None:
        coords = make_coordinates(img, right_line)
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), color_right, thickness)
    return img


def rail_lane_detection(image_path):
    print("🔍 Langkah 1: Deteksi tepi rel dengan Canny...")
    edges = canny_edge_detection(image_path, 50, 150)

    print("🚄 Langkah 2: Deteksi garis rel menggunakan Hough Transform...")
    img_color = cv2.imread(image_path)
    cropped_edges = region_of_interest(edges)

    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=100,
        maxLineGap=5
    )

    line_image = np.zeros_like(img_color)
    if lines is not None:
        averaged_lines = average_slope_intercept(lines)
        draw_lines(line_image, averaged_lines)

    # ✨ Kombinasi 3 layer: gambar asli + Canny + garis
    blend_edges = cv2.addWeighted(img_color, 0.8, edges_colored, 0.3, 0)
    combined = cv2.addWeighted(blend_edges, 1, line_image, 1, 1)

    # Simpan hasil final
    output_path = "combined_detection.jpg"
    cv2.imwrite(output_path, combined)
    print(f"✅ Hasil akhir disimpan sebagai '{output_path}'")

    # Tampilkan hasil di jendela OpenCV
    cv2.imshow("🚆 Hasil Deteksi Jalur Rel (Canny + Garis)", combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = r"D:\Tugas Mei\semester 7\Prak. Kontrol Cerdas\week6\rail_segmentation\test\images\1000195092_0014-0_jpeg.rf.bdcc36fa5199304696bb201c6db9c07c.jpg"
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"⚠️ File tidak ditemukan di: {image_path}")

    rail_lane_detection(image_path)
