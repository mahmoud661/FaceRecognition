import tkinter as tk
from tkinter import ttk
from functions import capture_and_train, recognize_faces
import os

if not os.path.exists('dataset'):
    os.makedirs('dataset')
if not os.path.exists('trainer'):
    os.makedirs('trainer')  # Create the main application window
root = tk.Tk()
root.title("Face Recognition System")
root.geometry("700x500")
root.configure(bg="#0a0a0a")  # Darker background for a sleek look

# Style for buttons
style = ttk.Style()

style.configure("TButton",
                font=("Helvetica", 16, "bold"),
                foreground="#f1c40f",
                background="#0a0a0a",
                padding=15,
                borderwidth=1,
                focusthickness=0,
                relief="flat")

style.map("TButton",
          background=[("active", "#0a0a0a"), ("!disabled", "#0a0a0a")],
          foreground=[("pressed", "#f1c40f"), ("active", "#f1c40f")])

# Title label
label = ttk.Label(root, text="Face Recognition System", font=("Helvetica", 24, "bold"),
                  foreground="#f1c40f", background="#0a0a0a")
label.pack(pady=20)

# Buttons for capturing/training and recognition
capture_button = ttk.Button(
    root, text="Capture & Train Faces", command=capture_and_train, style="TButton")
capture_button.pack(pady=30)

recognize_button = ttk.Button(
    root, text="Recognize Faces", command=recognize_faces, style="TButton")
recognize_button.pack(pady=30)

# Start the tkinter main loop
root.mainloop()
