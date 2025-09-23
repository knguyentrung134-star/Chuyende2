import cv2
import tkinter as tk
from tkinter import filedialog

# ===== Load Models =====
faceNet = cv2.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
ageNet = cv2.dnn.readNet("age_net.caffemodel", "age_deploy.prototxt")
genderNet = cv2.dnn.readNet("gender_net.caffemodel", "gender_deploy.prototxt")

# ===== Config =====
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# ===== Resize giữ nguyên tỷ lệ ảnh =====
def resizeWithAspectRatio(image, max_width=800, max_height=600):
    """
    Hàm resize ảnh giữ nguyên tỷ lệ gốc,
    fit vừa trong khung max_width x max_height
    """
    h, w = image.shape[:2]
    scale = min(max_width / w, max_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

# ===== Detect face =====
def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            # Vẽ khung khuôn mặt
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2, 8)
    return frameOpencvDnn, bboxes

# ===== Predict Age + Gender =====
def predictAgeGender(face):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]

    return gender, age

# ===== Video Mode =====
def processVideo():
    cap = cv2.VideoCapture(0)
    padding = 20
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break
        resultImg, bboxes = getFaceBox(faceNet, frame)
        for bbox in bboxes:
            face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                         max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
            gender, age = predictAgeGender(face)
            label = f"{gender}, {age}"
            cv2.putText(resultImg, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Resize trước khi show
        resized = resizeWithAspectRatio(resultImg)
        cv2.imshow("Video - Age & Gender", resized)

        if cv2.waitKey(1) == 27:  # ESC để thoát
            break
    cap.release()
    cv2.destroyAllWindows()

# ===== Image Mode with Button =====
def processImage():
    def chooseFile():
        file_path = filedialog.askopenfilename(title="Chọn ảnh",
                                               filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return
        frame = cv2.imread(file_path)
        if frame is None:
            print("Không thể đọc ảnh!")
            return

        resultImg, bboxes = getFaceBox(faceNet, frame)
        padding = 20
        for bbox in bboxes:
            face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
                         max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]
            gender, age = predictAgeGender(face)
            label = f"{gender}, {age}"
            cv2.putText(resultImg, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Resize giữ nguyên tỷ lệ trước khi show
        resized = resizeWithAspectRatio(resultImg)
        cv2.imshow("Image - Age & Gender", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    root = tk.Tk()
    root.title("Chọn ảnh nhận diện tuổi & giới tính")
    root.geometry("300x150")

    btn = tk.Button(root, text="Chọn ảnh", command=chooseFile, height=2, width=15, bg="lightblue")
    btn.pack(pady=20)

    root.mainloop()

# ===== Menu =====
print("Chọn chế độ:")
print("1 - Nhận diện từ Webcam")
print("2 - Nhận diện từ Ảnh (Upload)")
choice = input("Nhập lựa chọn: ")

if choice == "1":
    processVideo()
elif choice == "2":
    processImage()
else:
    print("Lựa chọn không hợp lệ!")
