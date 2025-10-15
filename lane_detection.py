import numpy as np
import cv2

def region_of_interest(img):
    height, width = img.shape[:2]
    polygons = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.6), int(height * 0.65)),
        (int(width * 0.4), int(height * 0.65))
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

def Hough_transform(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)  

    cropped_edges = region_of_interest(edges)
    lines = cv2.HoughLinesP(
    cropped_edges,
    rho=1,
    theta=np.pi/180,
    threshold=80,
    minLineLength=100,
    maxLineGap=5
    )


    line_image = np.zeros_like(image)
    if lines is not None:
        averaged_lines = average_slope_intercept(lines)
        draw_lines(line_image, averaged_lines)

    combined = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combined

if __name__ == "__main__":
    image_path = r"D:\Tugas Mei\semester 7\Prak. Kontrol Cerdas\week6\rail_segmentation\test\images\1000195092_0014-0_jpeg.rf.bdcc36fa5199304696bb201c6db9c07c.jpg"
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("⚠️ Gambar tidak ditemukan, pastikan path-nya benar!")

    print("🔍 Mendeteksi jalur rel...")
    result = Hough_transform(img)

    cv2.imshow("🚄 Hasil Deteksi Jalur Rel (Presisi Tinggi)", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("hasil_deteksi_jalur_rel.jpg", result)
    print("✅ Hasil disimpan sebagai 'hasil_deteksi_jalur_rel.jpg'")
