import os
import cv2
import numpy as np
from PIL import Image
from tkinter import simpledialog, messagebox
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from DB import insert_user, fetch_all_users, user_exists, insert_face_embedding, fetch_all_user_embeddings

# Load the DNN-based face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Load the pre-trained MobileNetV2 model for embeddings
embedding_model = hub.load(
    "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4")


def ensure_directories_exist():
    """Ensure that the dataset and trainer directories exist."""
    if not os.path.exists('dataset'):
        os.makedirs('dataset')


def normalize_embedding(embedding):
    """Normalize the embedding to have a unit norm."""
    return embedding / np.linalg.norm(embedding)


def get_face_embedding(image):
    """Extract face embeddings using TensorFlow Hub (MobileNetV2)."""
    # Resize the image to 224x224 as required by MobileNetV2
    image = image.resize((224, 224))

    # Preprocess the image
    img_array = keras_image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get embeddings
    embeddings = embedding_model(img_array)
    normalized_embedding = embeddings.numpy()[0]  # Normalize the embedding
    return normalize_embedding(normalized_embedding)


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

                # Extract face ROI and convert it to RGB for embedding extraction
                face_img = img[y:y1, x:x1]
                pil_img = Image.fromarray(
                    cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                embedding = get_face_embedding(pil_img)

                if embedding is not None:
                    # Save the embedding to the DB
                    insert_face_embedding(face_name, embedding)
                    print(f"Captured face {count} for {face_name}")

        if count >= 50:  # Capture 50 face samples
            break
        k = cv2.waitKey(100) & 0xff
        if k == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    print(f"\n[INFO] Training completed for {
          face_name}. {count} faces captured.")


def recognize_faces():
    users = fetch_all_users()  # Fetch users from the database
    # Fetch all embeddings from the database
    embeddings_db = fetch_all_user_embeddings()

    if not embeddings_db:
        messagebox.showerror(
            "Error", "No face embeddings found in the database! Please capture and train faces first.")
        return

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
            if confidence > 0.5:  # Face detection confidence
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x1, y1) = box.astype("int")
                cv2.rectangle(img, (x, y), (x1, y1), (0, 255, 0), 2)

                # Extract face ROI and convert it to RGB for embedding extraction
                face_img = img[y:y1, x:x1]
                pil_img = Image.fromarray(
                    cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                embedding = get_face_embedding(pil_img)

                if embedding is not None:
                    print(f"[DEBUG] Extracted embedding for recognition: {
                          embedding}")

                    # Compute distances between the detected face and known embeddings
                    distances = {
                        user_id: np.linalg.norm(normalize_embedding(
                            embedding) - normalize_embedding(emb))
                        for user_id, emb in embeddings_db.items()
                    }

                    print(f"[DEBUG] Calculated distances: {distances}")

                    # Find the closest user based on minimum distance
                    closest_id = min(distances, key=distances.get)
                    closest_distance = distances[closest_id]

                    print(f"[DEBUG] Closest match: User ID {
                          closest_id}, Distance: {closest_distance}")

                    # Show the recognized user's name and the confidence level
                    name = users.get(closest_id, "Unknown")
                    cv2.putText(img, str(closest_id), (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img, f"{100 - (closest_distance * 100):.2f}%",
                                (x + 5, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1)

        # Show the resulting frame with recognized faces
        cv2.imshow('camera', img)

        if cv2.getWindowProperty('camera', cv2.WND_PROP_VISIBLE) < 1:
            break

        k = cv2.waitKey(10) & 0xff
        if k == 27:  # Press 'ESC' to exit
            break

    cam.release()
    cv2.destroyAllWindows()
