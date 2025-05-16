import os
import cv2
import numpy as np
from ultralytics import YOLO

# โหลดโมเดลที่ฝึกไว้
model = YOLO('./VscodeFordetect/best.pt')

# ระบุโฟลเดอร์ที่มีภาพหรือวิดีโอ
image_dir = './VscodeFordetect/video'

for file_name in os.listdir(image_dir):
    file_path = os.path.join(image_dir, file_name)

    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        print(f"Processing image: {file_path}...")
        results = model.predict(source=file_path, save=False)
        img = cv2.imread(file_path)
        img_with_edges = img.copy()

        if results[0].masks is None:
            print("No objects detected in the image!")
        else:
            masks = results[0].masks.data.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy()
            water_class_id = 3
            water_mask_index = None
            for i, cls_id in enumerate(class_ids):
                if int(cls_id) == water_class_id:
                    water_mask_index = i
                    break

            if water_mask_index is None:
                print("Water (class 3) not detected in the image!")
            else:
                water_mask = masks[water_mask_index].astype(np.uint8)
                height, width = img.shape[:2]
                water_mask_resized = cv2.resize(water_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                kernel = np.ones((5, 5), np.uint8)
                water_mask_cleaned = cv2.morphologyEx(water_mask_resized, cv2.MORPH_CLOSE, kernel)
                water_mask_cleaned = cv2.morphologyEx(water_mask_cleaned, cv2.MORPH_OPEN, kernel)
                edges = cv2.Canny(water_mask_cleaned * 255, 50, 150)
                img_with_edges[edges != 0] = [0, 0, 255]

        # แสดงภาพในหน้าต่างป๊อปอัพ
        cv2.imshow('Water-Sand Boundary', img_with_edges)
        cv2.waitKey(2000)  # แสดง 2 วินาที (2000 มิลลิวินาที)
        cv2.destroyWindow('Water-Sand Boundary')

    elif file_name.endswith(('.mp4', '.avi','.mkv')):
        print(f"Processing video: {file_path}...")
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {file_path}")
            continue

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

            # แสดงเฟรมในหน้าต่างป๊อปอัพ
            cv2.imshow('Water-Sand Boundary', frame_with_edges)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # กด 'q' เพื่อออก
                break

        cap.release()
        cv2.destroyAllWindows()

cv2.destroyAllWindows() 