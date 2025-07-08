import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

# Load and encode known faces
path = 'photos'
images = []
known_face_names = []
image_files = os.listdir(path)

for file in image_files:
    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
        img = cv2.imread(f'{path}/{file}')
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb_img)
        if len(encodings) > 0:
            images.append(encodings[0])
            known_face_names.append(os.path.splitext(file)[0])
        else:
            print(f"⚠️ No face found in image: {file}")
    else:
        print(f"⛔ Skipping file: {file}")

# Create CSV for attendance
now = datetime.now()
date = now.strftime("%Y-%m-%d")
csv_file = open(f'{date}.csv', 'w+', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Name', 'Time'])

# Track marked students
marked = set()

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("❌ Failed to grab frame from camera.")
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(images, face_encoding)
        face_distances = face_recognition.face_distance(images, face_encoding)
        match_index = np.argmin(face_distances)

        name = "Unknown"
        if matches[match_index]:
            name = known_face_names[match_index]

            if name not in marked:
                marked.add(name)
                time_str = datetime.now().strftime("%H:%M:%S")
                csv_writer.writerow([name, time_str])
                print(f"✅ {name} marked present at {time_str}")

        # Draw rectangle and label
        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('Face Recognition Attendance System', frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
csv_file.close()
