# from ultralytics import YOLO
# import cv2

# class_map = {
#     "car": "o_to",
#     "bus": "xe_buyt",
#     "truck": "xe_tai",
#     "motorbike": "xe_may",
# }

# model = YOLO("yolov11s.pt")


# img_path = "test.jpg"
# results = model(img_path)

# img = cv2.imread(img_path)

# for r in results:
#     boxes = r.boxes
#     names = r.names  

#     for box in boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         cls_id = int(box.cls[0])
#         cls_name = names[cls_id]

#         vn_name = class_map.get(cls_name, cls_name)

#         cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(img, vn_name, (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# cv2.imwrite("result.jpg", img)
# print("Da luu anh: result.jpg")


# from ultralytics import YOLO
# import cv2
# from tkinter import Tk, filedialog

# root = Tk()
# root.withdraw()
# image_path = filedialog.askopenfilename(
#     title="Chọn ảnh để nhận diện",
#     filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
# )

# if not image_path:
#     print(".")
#     exit()

# print(f" ...: {image_path}")


# model = YOLO("runs/detect/train5/weights/best.pt")  

# results = model.predict(source=image_path, conf=0.5)


# for r in results:
#     img = r.plot()  
#     cv2.imshow("...", img)
#     cv2.imwrite("result_detect.jpg", img)  
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# print("...: result_detect.jpg")

from ultralytics import YOLO
import cv2
from tkinter import Tk, filedialog
import os
from collections import Counter

model = YOLO("runs/detect/train2/weights/best.pt")

root = Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Chọn ảnh hoặc video",
    filetypes=[
        ("Images", "*.jpg *.jpeg *.png"),
        ("Videos", "*.mp4 *.avi *.mov")
    ]
)

if not file_path:
    print("Chưa chọn file!")
    exit()

ext = os.path.splitext(file_path)[1].lower()

if ext in [".jpg", ".jpeg", ".png"]:
    results = model.predict(source=file_path, conf=0.5)

    r = results[0]
    boxes = r.boxes

    class_ids = boxes.cls.cpu().numpy().astype(int)
    counts = Counter(class_ids)

    print(" SỐ LƯỢNG PHƯƠNG TIỆN:")
    total = 0
    for cls_id, num in counts.items():
        name = model.names[cls_id]
        print(f" - {name}: {num}")
        total += num

    print(f" Tổng số phương tiện: {total}")

    img = r.plot()

    y = 30
    for cls_id, num in counts.items():
        text = f"{model.names[cls_id]}: {num}"
        cv2.putText(img, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30

    cv2.imshow("Kết quả (Ảnh)", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

elif ext in [".mp4", ".avi", ".mov"]:
    cap = cv2.VideoCapture(file_path)

    if not cap.isOpened():
        print(" Không mở được video!")
        exit()

    print(" Nhận diện & đếm phương tiện (ESC để thoát)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.5, verbose=False)
        r = results[0]

        boxes = r.boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        counts = Counter(class_ids)

        annotated_frame = r.plot()

        y = 30
        total = 0
        for cls_id, num in counts.items():
            text = f"{model.names[cls_id]}: {num}"
            cv2.putText(annotated_frame, text, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y += 30
            total += num

        cv2.putText(annotated_frame, f"Total: {total}", (10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        annotated_frame = cv2.resize(annotated_frame, (960, 540))
        cv2.imshow("Kết quả (Video)", annotated_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

else:
    print(" File không được hỗ trợ!")