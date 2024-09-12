import os
import cv2
import numpy as np
from PIL import Image
from tkinter import simpledialog, messagebox
from DB import insert_user, fetch_all_users, user_exists

# Load the DNN-based face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)


def ensure_directories_exist():
    """Ensure that the dataset and trainer directories exist."""
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    if not os.path.exists('trainer'):
        os.makedirs('trainer')


def capture_and_train():
    ensure_directories_exist()  # Ensure directories are present

    face_name = simpledialog.askstring("Input", "Enter the name:")
    if not face_name:
        messagebox.showerror("Error", "You must enter a name!")
        return

    if user_exists(face_name):
        messagebox.showinfo("Info", "User already exists!")
        return

    face_id = insert_user(face_name)
    if not face_id:
        messagebox.showerror(
            "Error", "Failed to insert user into the database!")
        return

    print(f"\n[INFO] Initializing face capture for {
          face_name}. Look at the camera and wait...")

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    count = 0
    while True:
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [
                                     104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (255, 0, 0), 2)
                count += 1
                gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y:y1, x:x1]
                cv2.imwrite(
                    f"dataset/User.{face_id}.{str(count)}.jpg", gray_face)

        if count >= 50:
            break
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    print(f"\n[INFO] Training faces for {face_name}. Please wait...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, ids = getImagesAndLabels('dataset')
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')
    print(f"\n[INFO] {len(np.unique(ids))} faces trained. Exiting Program.")


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(img_numpy)
        ids.append(id)

    return faceSamples, ids


def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    try:
        recognizer.read('trainer/trainer.yml')
    except:
        messagebox.showerror(
            "Error", "No trained model found! Please capture and train faces first.")
        return

    users = fetch_all_users()

    cam = cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)

    while True:
        ret, img = cam.read()
        if not ret:
            break
        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [
                                     104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)

                gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y:y1, x:x1]

                id, pred_confidence = recognizer.predict(gray_face)

                threshold = 60

                if (100 - round(pred_confidence)) > threshold:
                    name = users.get(id, "Unknown")
                    confidence_text = f"   {100 - round(pred_confidence)}%"
                else:
                    name = "Unknown"
                    confidence_text = "  Unknown"

                cv2.putText(img, str(name), (x + 5, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(img, str(confidence_text), (x + 5, y1 + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) < 1:
            break

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
