

import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam. Please allow camera access in System Settings.")
    exit()

ret, frame = cap.read()
cap.release()

if ret:
    print("✅ Camera works!")
else:
    print("❌ Failed to capture image.")