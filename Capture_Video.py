import cv2
import tensorflow as tf
import numpy as np
import os
from collections import deque, Counter


def start_realtime_emotion_detection(
    model_path=os.path.join(os.path.dirname(__file__), "FER_Model.keras"),
):

    if not os.path.exists(model_path):
        print("❌ Model file not found.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully")

    labels = [
        "angry",
        "contempt",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "suprise",
    ]

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("Press 'q' to exit.")

    prediction_buffer = deque(maxlen=10)
    stable_label = ""
    confidence = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_frame, scaleFactor=1.3, minNeighbors=5
        )

        for x, y, w, h in faces:
            face = frame[y : y + h, x : x + w]
            face = cv2.resize(face, (128, 128))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face.astype("float32")
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)

            max_idx = np.argmax(prediction)
            prediction_buffer.append(max_idx)

            if len(prediction_buffer) == prediction_buffer.maxlen:
                most_common = Counter(prediction_buffer).most_common(1)[0][0]
                stable_label = labels[most_common]
                confidence = float(np.max(prediction)) * 100

            color = (0, 255, 0) if stable_label == "Happy" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            if stable_label != "":
                cv2.putText(
                    frame,
                    f"{stable_label} ({confidence:.1f}%)",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

        cv2.imshow("Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


start_realtime_emotion_detection()
