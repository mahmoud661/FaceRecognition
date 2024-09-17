import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk  # Importing PIL to display placeholder image
from functions import capture_and_train, capture_and_recognize_face
from DB import create_table_if_not_exists

# Dark mode colors
DARK_BG = "#111111"
BUTTON_BG = "#1e1e1e"
BUTTON_ACTIVE_BG = "#2e2e2e"
TEXT_COLOR = "#ffffff"
CAMERA_WIDTH = 640  # Adjusted for smaller size
CAMERA_HEIGHT = 480

def on_capture_and_train(video_label, root):
    """Handler for the 'Capture and Train' button."""
    face_name = simpledialog.askstring("Input", "Enter the name:")
    if not face_name:
        messagebox.showerror("Error", "You must enter a name!")
        return
    capture_and_train(face_name, video_label, root)

def on_recognize_face(video_label, root):
    """Handler for the 'Recognize Face' button."""
    capture_and_recognize_face(video_label, root)

def main():
    create_table_if_not_exists()

    root = tk.Tk()
    root.title("Face Recognition System")

    # Set window background color
    root.configure(bg=DARK_BG)

    # Create a frame for the camera and buttons
    main_frame = tk.Frame(root, bg=DARK_BG)
    main_frame.grid(row=0, column=0, padx=20, pady=20)

    # Load a placeholder image for the camera space
    placeholder_img = Image.new('RGB', (CAMERA_WIDTH, CAMERA_HEIGHT), color="#383838")
    placeholder_img_tk = ImageTk.PhotoImage(placeholder_img)

    # Camera space (Placeholder for the camera feed)
    video_label = tk.Label(main_frame, image=placeholder_img_tk, bg="#383838")
    video_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    # Create buttons with dark mode styling and use grid layout
    capture_btn = tk.Button(
        main_frame,
        text="Capture and Train",
        command=lambda: on_capture_and_train(video_label, root),
        bg=BUTTON_BG,
        fg=TEXT_COLOR,
        activebackground=BUTTON_ACTIVE_BG,
        activeforeground=TEXT_COLOR,
        bd=0,
        padx=10,
        pady=5
    )
    capture_btn.grid(row=1, column=0, padx=10, pady=10)

    recognize_btn = tk.Button(
        main_frame,
        text="Recognize Face",
        command=lambda: on_recognize_face(video_label, root),
        bg=BUTTON_BG,
        fg=TEXT_COLOR,
        activebackground=BUTTON_ACTIVE_BG,
        activeforeground=TEXT_COLOR,
        bd=0,
        padx=10,
        pady=5
    )
    recognize_btn.grid(row=1, column=1, padx=10, pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()
