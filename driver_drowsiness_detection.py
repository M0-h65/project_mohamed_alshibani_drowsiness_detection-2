import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
import pygame
import threading
import tkinter as tk
from PIL import Image, ImageTk

# إعدادات العتبات
EAR_THRESHOLD = 0.2
EAR_CONSEC_FRAMES = 20
MAR_THRESHOLD = 0.6
MAR_CONSEC_FRAMES = 15

# نقاط العين والفم حسب Mediapipe
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
# نقاط الفم: 13 (أسفل منتصف الفم), 14 (أعلى منتصف الفم), 78 (يسار الفم), 308 (يمين الفم)
MOUTH_IDX = [78, 308, 13, 14]

# تحميل الأصوات
pygame.mixer.init()
pygame.mixer.music.load("alert.wav")  # ضع ملف صوت باسم alert.wav في نفس المجلد

def sound_alert():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

def eye_aspect_ratio(eye):
    # حساب EAR
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # mouth[2]: أسفل منتصف الفم (13)
    # mouth[3]: أعلى منتصف الفم (14)
    # mouth[0]: يسار الفم (78)
    # mouth[1]: يمين الفم (308)
    A = dist.euclidean(mouth[2], mouth[3])  # أعلى الفم وأسفله
    C = dist.euclidean(mouth[0], mouth[1])  # طرفي الفم
    mar = A / C
    return mar

class DrowsinessDetector:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.eye_counter = 0
        self.mouth_counter = 0
        self.drowsy = False
        self.yawn = False
        self.running = True

        # Tkinter GUI
        self.root = tk.Tk()
        self.root.title("Driver Drowsiness Detection")
        self.label = tk.Label(self.root)
        self.label.pack()
        self.status_label = tk.Label(self.root, text="Status: Awake", font=('Arial', 16), fg="green")
        self.status_label.pack()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_thread = threading.Thread(target=self.video_loop)
        self.video_thread.start()
        self.root.mainloop()

    def on_close(self):
        self.running = False
        self.cap.release()
        self.root.destroy()

    def video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(frame_rgb)
            status_text = "Awake"
            status_color = "green"

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame.shape
                    # نقاط العين اليسرى واليمنى
                    left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE_IDX]
                    right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE_IDX]
                    # نقاط الفم
                    mouth = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in MOUTH_IDX]

                    # حساب EAR و MAR
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0
                    mar = mouth_aspect_ratio(mouth)

                    # رسم النقاط
                    for (x, y) in left_eye + right_eye + mouth:
                        cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

                    # منطق النعاس - العين
                    if ear < EAR_THRESHOLD:
                        self.eye_counter += 1
                        if self.eye_counter >= EAR_CONSEC_FRAMES:
                            status_text = "Drowsy!"
                            status_color = "red"
                            self.drowsy = True
                            sound_alert()
                    else:
                        self.eye_counter = 0
                        self.drowsy = False

                    # منطق التثاؤب - الفم
                    if mar > MAR_THRESHOLD:
                        self.mouth_counter += 1
                        if self.mouth_counter >= MAR_CONSEC_FRAMES:
                            status_text = "Yawning!"
                            status_color = "orange"
                            self.yawn = True
                            sound_alert()
                    else:
                        self.mouth_counter = 0
                        self.yawn = False

                    # عرض القيم على الإطار
                    cv2.putText(frame, f'EAR: {ear:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    cv2.putText(frame, f'MAR: {mar:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            # تحديث واجهة المستخدم
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
            self.status_label.configure(text=f"Status: {status_text}", fg=status_color)

        self.cap.release()

if __name__ == "__main__":
    DrowsinessDetector()