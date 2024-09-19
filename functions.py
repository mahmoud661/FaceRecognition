from PIL import ImageTk
import cv2
import numpy as np
import face_recognition
from PIL import Image
import os
from DB import insert_user, insert_face_embedding, fetch_all_user_embeddings, user_exists
from tkinter import simpledialog, messagebox

# Load the DNN-based face detection model
modelFile = "res10_300x300_ssd_iter_140000.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Cache for known face encodings
known_face_encodings = {}
face_names = {}


def load_known_face_encodings():
    """Load known face encodings from the database and cache them."""
    global known_face_encodings
    global face_names
    known_face_encodings = fetch_all_user_embeddings()
    face_names = {name: encoding for name,
                  encoding in known_face_encodings.items()}
    print("Loaded known face encodings from database.")


def get_face_embedding(image):
    """Extract face embeddings using face_recognition library."""
    image = np.array(image)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings


def save_face_image(face_img, face_name, count):
    """Save face image to disk."""
    if not os.path.exists('images'):
        os.makedirs('images')
    image_path = f"images/{face_name}_{count}.jpg"
    cv2.imwrite(image_path, face_img)
    return image_path


def add_new_face(name, image):
    """Add a new face to the database."""
    embeddings = get_face_embedding(image)
    if embeddings:
        embedding = embeddings[0]  # Assume the first face found
        if not user_exists(name):
            user_id = insert_user(name)
            insert_face_embedding(name, np.array(embedding))
            # Reload known face encodings after adding new face
            load_known_face_encodings()
            print(f"Added new face for {name}.")
        else:
            print(f"User {name} already exists.")
    else:
        print("No face found in the image.")


def recognize_face(face_encoding):

    best_match_name = "Unknown"
    best_match_distance = float('inf')

    for name, known_encoding in face_names.items():
        known_encoding = np.array(known_encoding)
        match = face_recognition.compare_faces([known_encoding], face_encoding)

        if match[0]:

            face_distance = face_recognition.face_distance([known_encoding], face_encoding)[0]


            if face_distance < best_match_distance:
                best_match_distance = face_distance
                best_match_name = name

    return best_match_name

def capture_and_train(face_name, video_label, root):
    """Capture multiple images for a new user, extract embeddings, and save them to the database."""
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
    user_embeddings = []

    def show_frame():
        nonlocal count, user_embeddings

        ret, img = cam.read()
        if not ret:
            return

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

                # Extract embedding for both original and augmented images
                embedding = get_face_embedding(pil_img)

                # Save embeddings for averaging later
                if embedding:
                    user_embeddings.append(embedding[0])

        # Update the video frame in the label
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        video_label.img_tk = img_tk  # keep a reference to prevent garbage collection
        video_label.config(image=img_tk)

        # Stop after capturing 50 face samples
        if count < 50:
            root.after(10, show_frame)
        else:
            cam.release()
            avg_embedding = np.mean(user_embeddings, axis=0)
            insert_face_embedding(face_name, avg_embedding)
            messagebox.showinfo("Info", f"Training completed for {
                                face_name}. {count} faces captured.")

    show_frame()


def capture_and_recognize_face(video_label, root):
    """Capture a face using the webcam and recognize it using DNN for detection and face_recognition for recognition."""
    load_known_face_encodings()  # Load known face encodings once at the start
    cam = cv2.VideoCapture(0)

    def show_frame():
        ret, frame = cam.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # Use DNN to detect faces
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()

        face_locations = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:  # Only consider detections with high confidence
                # Compute the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                (x, y, x1, y1) = box.astype("int")

                # Add the bounding box to the face_locations list (in top, right, bottom, left format)
                face_locations.append((y, x1, y1, x))

                # Draw the bounding box on the frame
                cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 0), 2)

        # If faces were detected by DNN, use face_recognition to encode them
        if face_locations:
            face_encodings = face_recognition.face_encodings(frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Recognize the face using existing face recognition method
                name = recognize_face(face_encoding)

                # Display the name on the frame
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Update the video frame in the label
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        video_label.img_tk = img_tk  # keep a reference to prevent garbage collection
        video_label.config(image=img_tk)

        # Repeat the loop every 10ms
        root.after(10, show_frame)

    show_frame()
