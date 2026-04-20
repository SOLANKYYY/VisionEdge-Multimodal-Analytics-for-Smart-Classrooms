import cv2
from deepface import DeepFace
import threading
import time

# --- CONFIGURATION ---
# We analyze the frame every 10 frames to keep the FPS high
FRAME_SKIP = 10 
# Scale down the image for the AI (0.5 = 50% smaller)
AI_SCALE = 0.5 

class EmotionScanner:
    def __init__(self):
        self.latest_results = []
        self.is_processing = False

    def scan(self, frame):
        if self.is_processing:
            return
        
        # Run AI in a separate thread so it doesn't lag the video
        thread = threading.Thread(target=self._run_inference, args=(frame,))
        thread.daemon = True
        thread.start()

    def _run_inference(self, frame):
        self.is_processing = True
        try:
            # Resize and analyze
            small_frame = cv2.resize(frame, (0, 0), fx=AI_SCALE, fy=AI_SCALE)
            
            # Use 'opencv' backend for speed
            results = DeepFace.analyze(
                small_frame, 
                actions=['emotion'], 
                enforce_detection=False,
                detector_backend='opencv' 
            )
            self.latest_results = results
        except Exception:
            pass
        self.is_processing = False

# Initialize
cap = cv2.VideoCapture(0)
scanner = EmotionScanner()
counter = 0

print("VisionEdge: High-Performance Mode Active")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Trigger AI every few frames
    if counter % FRAME_SKIP == 0:
        scanner.scan(frame)

    # 2. Draw the LATEST saved results (don't wait for AI to finish)
    if scanner.latest_results:
        for face in scanner.latest_results:
            # Scale coordinates back up to original size
            x = int(face['region']['x'] / AI_SCALE)
            y = int(face['region']['y'] / AI_SCALE)
            w = int(face['region']['w'] / AI_SCALE)
            h = int(face['region']['h'] / AI_SCALE)
            
            emotion = face['dominant_emotion']
            
            # Draw Box & Label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion.upper(), (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # 3. Show standard UI
    cv2.putText(frame, "STATUS: SMOOTH", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imshow('VisionEdge - Threaded AI', frame)

    counter += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
