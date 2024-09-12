import tkinter as tk
from tkinter import ttk
from functions import capture_and_train, recognize_faces, ensure_directories_exist

# Create the main application window
root = tk.Tk()
root.title("Face Recognition System")


# Create a frame for buttons
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Define buttons and their commands
train_button = ttk.Button(
    frame, text="Capture and Train", command=capture_and_train)
train_button.grid(row=0, column=0, padx=5, pady=5)

recognize_button = ttk.Button(
    frame, text="Recognize Faces", command=recognize_faces)
recognize_button.grid(row=1, column=0, padx=5, pady=5)

# Start the GUI event loop
root.mainloop()
