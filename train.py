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


from ultralytics import YOLO
import cv2
from tkinter import Tk, filedialog

root = Tk()
root.withdraw()
image_path = filedialog.askopenfilename(
    title="Chọn ảnh để nhận diện",
    filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp")]
)

if not image_path:
    print(".")
    exit()

print(f" ...: {image_path}")


model = YOLO("runs/detect/train5/weights/best.pt")  

results = model.predict(source=image_path, conf=0.5)


for r in results:
    img = r.plot()  
    cv2.imshow("...", img)
    cv2.imwrite("result_detect.jpg", img)  
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("...: result_detect.jpg")