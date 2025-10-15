from ultralytics import YOLO
import cv2

# Load model hasil training (path lengkap)
model = YOLO(r"D:\Tugas Mei\semester 7\Prak. Kontrol Cerdas\week6\rail_segmentation\runs\segment\train6\weights\best.pt")

# Predict folder gambar uji
results = model.predict(
    source=r"D:\Tugas Mei\semester 7\Prak. Kontrol Cerdas\week6\rail_segmentation\test\images",  # folder gambar test
    show=True,   # tampilkan hasil di jendela
    save=True,   # simpan hasil ke folder runs/segment/predict/
    conf=0.5     # confidence threshold (0.5 artinya cukup yakin)
)

# Hilangkan kotak, hanya tampilkan mask
for r in results:
    img = r.plot(labels=False, boxes=False)  # hanya segmen rel
    cv2.imshow("Rail Segmentation Only", img)
    cv2.waitKey(0)
