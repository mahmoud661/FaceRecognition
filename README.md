# Face Recognition System

## Overview

This Face Recognition System is a Python-based application that allows users to capture and recognize faces using a webcam. The system uses OpenCV for video capture, `face_recognition` for face recognition tasks, and PostgreSQL for database storage. The GUI is built using `tkinter` and supports a dark mode theme.

## Features

- **Capture and Train**: Capture multiple images of a new user, extract face embeddings, and save them to the database.
- **Recognize Face**: Recognize and identify faces using the webcam.
- **Database Integration**: Uses PostgreSQL to store user data and face embeddings.

## Requirements

- Python 3.7 or higher
- `Pillow` for image processing
- `opencv-python` for video capture and face detection
- `face_recognition` for face recognition
- `numpy` for numerical operations
- `psycopg2` for PostgreSQL database connectivity

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/face-recognition-system.git
   cd face-recognition-system
