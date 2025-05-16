import os
import cv2
import numpy as np
from ultralytics import YOLO

# โหลดโมเดล YOLOv8
model = YOLO('./VscodeFordetect/best.pt')

# ระบุโฟลเดอร์ที่มีไฟล์ในเครื่อง
image_dir = './VscodeFordetect/video'

# URL ของสตรีมวิดีโอ (ตัวอย่างเช่น RTSP หรือ HLS)
# ถ้าคุณหา URL จริงจาก https://coastalradar.gistda.or.th ได้ ให้แทนที่ที่นี่
# ตัวอย่าง: stream_url = 'rtsp://[ip-address]/stream' หรือ 'https://.../stream.m3u8'
stream_url = 'https://coastalradar.gistda.or.th/cctvlive/120221903105469583224681635448311880059/hls/S057/playlist.m3u8'

# ตรวจสอบว่าโฟลเดอร์มีอยู่จริงหรือไม่ (สำหรับไฟล์ในเครื่อง)
if not os.path.exists(image_dir):
    print(f"Error: Directory {image_dir} does not exist!")
    exit()

# ฟังก์ชันประมวลผลเฟรม (ใช้ได้ทั้งไฟล์และสตรีม)
def process_frame(frame):
    frame_with_edges = frame.copy()
    results = model.predict(source=frame, save=False)

    if results[0].masks is not None:
        masks = results[0].masks.data.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()
        water_class_id = 3
        water_mask_index = None
        for i, cls_id in enumerate(class_ids):
            if int(cls_id) == water_class_id:
                water_mask_index = i
                break

        if water_mask_index is not None:
            water_mask = masks[water_mask_index].astype(np.uint8)
            height, width = frame.shape[:2]
            water_mask_resized = cv2.resize(water_mask, (width, height), interpolation=cv2.INTER_NEAREST)
            kernel = np.ones((5, 5), np.uint8)
            water_mask_cleaned = cv2.morphologyEx(water_mask_resized, cv2.MORPH_CLOSE, kernel)
            water_mask_cleaned = cv2.morphologyEx(water_mask_cleaned, cv2.MORPH_OPEN, kernel)
            edges = cv2.Canny(water_mask_cleaned * 255, 50, 150)
            frame_with_edges[edges != 0] = [0, 255, 0]

    return frame_with_edges

# 1. ประมวลผลไฟล์ภาพในเครื่อง
# for file_name in os.listdir(image_dir):
#     file_path = os.path.join(image_dir, file_name)

#     if file_name.endswith(('.jpg', '.jpeg', '.png')):
#         print(f"Processing image: {file_path}...")
#         img = cv2.imread(file_path)
#         if img is None:
#             print(f"Error: Cannot load image {file_path}")
#             continue
        
#         img_with_edges = process_frame(img)
#         cv2.imshow('Wave Boundary Detection', img_with_edges)
#         cv2.waitKey(2000)
#         cv2.destroyWindow('Wave Boundary Detection')

#     elif file_name.endswith(('.mp4', '.avi', '.mkv')):
#         print(f"Processing video: {file_path}...")
#         cap = cv2.VideoCapture(file_path)
#         if not cap.isOpened():
#             print(f"Error: Cannot open video {file_path}")
#             continue

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame_with_edges = process_frame(frame)
#             cv2.imshow('Wave Boundary Detection', frame_with_edges)
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

# 2. ประมวลผลวิดีโอสตรีมจากแหล่งภายนอก
print(f"Processing stream: {stream_url}...")
cap = cv2.VideoCapture(stream_url)
if not cap.isOpened():
    print(f"Error: Cannot open stream {stream_url}")
    print("Please ensure the stream URL is correct and accessible (e.g., RTSP or HLS URL).")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Stream interrupted or ended. Retrying...")
            cap.release()
            cap = cv2.VideoCapture(stream_url)  # พยายามเชื่อมต่อใหม่
            continue

        frame_with_edges = process_frame(frame)
        cv2.imshow('Wave Boundary Detection (Stream)', frame_with_edges)
        if  0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()