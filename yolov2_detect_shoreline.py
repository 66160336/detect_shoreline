import os
import cv2
import numpy as np
from ultralytics import YOLO

# โหลดโมเดล YOLOv8 ที่ฝึกไว้
# ใช้พาธ './VscodeFordetect/best.pt' ซึ่งควรอยู่ในโฟลเดอร์ VscodeFordetect
# ตรวจสอบว่าไฟล์ best.pt อยู่ในพาธนี้จริงๆ
model = YOLO('./VscodeFordetect/best.pt')

# ระบุโฟลเดอร์ที่มีภาพหรือวิดีโอ
# './VscodeFordetect/video' ควรมีไฟล์ภาพ (.jpg, .png) หรือวิดีโอ (.mp4, .avi, .mkv)
image_dir = './VscodeFordetect/video'

# ตรวจสอบว่าโฟลเดอร์มีอยู่จริงหรือไม่
if not os.path.exists(image_dir):
    print(f"Error: Directory {image_dir} does not exist!")
    exit()

# วนลูปผ่านทุกไฟล์ในโฟลเดอร์
for file_name in os.listdir(image_dir):
    file_path = os.path.join(image_dir, file_name)

    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        print(f"Processing image: {file_path}...")
        # ทำนายผลด้วย YOLOv8
        results = model.predict(source=file_path, save=False)
        # โหลดภาพ
        img = cv2.imread(file_path)
        # ตรวจสอบว่าโหลดภาพได้หรือไม่
        if img is None:
            print(f"Error: Cannot load image {file_path}")
            continue
        
        img_with_edges = img.copy()

        # ตรวจสอบว่ามี mask หรือไม่
        if results[0].masks is None:
            print("No objects detected in the image!")
        else:
            masks = results[0].masks.data.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            water_class_id = 3  # จาก data.yaml: water = class 3
            water_mask_index = None
            
            # หา mask ของน้ำ (water)
            for i, cls_id in enumerate(class_ids):
                if int(cls_id) == water_class_id:
                    water_mask_index = i
                    break

            if water_mask_index is None:
                print("Water (class 3) not detected in the image!")
            else:
                # ประมวลผล mask
                water_mask = masks[water_mask_index].astype(np.uint8)
                height, width = img.shape[:2]
                water_mask_resized = cv2.resize(water_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                kernel = np.ones((5, 5), np.uint8)
                water_mask_cleaned = cv2.morphologyEx(water_mask_resized, cv2.MORPH_CLOSE, kernel)
                water_mask_cleaned = cv2.morphologyEx(water_mask_cleaned, cv2.MORPH_OPEN, kernel)
                edges = cv2.Canny(water_mask_cleaned * 255, 50, 150)
                # วาดขอบสีแดง
                img_with_edges[edges != 0] = [0, 0, 255]

        # แสดงภาพในหน้าต่างป๊อปอัพ
        cv2.imshow('Water-Sand Boundary', img_with_edges)
        cv2.waitKey(2000)  # รอ 2 วินาที
        cv2.destroyWindow('Water-Sand Boundary')

    elif file_name.endswith(('.mp4', '.avi', '.mkv')):
        print(f"Processing video: {file_path}...")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {file_path}")
            continue

        # เพิ่มการบันทึกวิดีโอผลลัพธ์
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f'output_{file_name}', fourcc, fps, (frame_width, frame_height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(source=frame, save=False)
            frame_with_edges = frame.copy()

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
                    frame_with_edges[edges != 0] = [0, 0, 255]

            # บันทึกเฟรมลงในวิดีโอผลลัพธ์
            out.write(frame_with_edges)

            # แสดงเฟรมในหน้าต่างป๊อปอัพ
            cv2.imshow('Water-Sand Boundary', frame_with_edges)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # กด 'q' เพื่อออก
                break

        cap.release()
        out.release()  # ปิดไฟล์วิดีโอผลลัพธ์
        cv2.destroyAllWindows()

cv2.destroyAllWindows()