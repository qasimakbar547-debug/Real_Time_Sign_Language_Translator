import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import av
import time

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Real-Time Sign Language Translator")
st.write("Show your hand sign in front of the camera.")

st.info("Supported Signs: A • B • C • D • E")


# =====================================================
# FILES
# =====================================================

MODEL_PATH = "sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"


# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load(MODEL_PATH)


# =====================================================
# MEDIAPIPE
# =====================================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# =====================================================
# VIDEO PROCESSOR
# =====================================================

class SignLanguageProcessor(VideoProcessorBase):

    def _init_(self):

        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=HAND_MODEL_PATH
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.landmarker = vision.HandLandmarker.create_from_options(
            options
        )

        self.timestamp = 0


    def recv(self, frame):

        # Camera frame
        img = frame.to_ndarray(format="bgr24")

        # Convert BGR → RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Timestamp
        self.timestamp += 33

        # MediaPipe image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # Detect hand
        result = self.landmarker.detect_for_video(
            mp_image,
            self.timestamp
        )

        prediction = "No Hand"
        confidence = 0.0

        # =================================================
        # HAND DETECTED
        # =================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            h, w, _ = img.shape

            # Draw landmarks
            for landmark in hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    img,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Draw connections
            for start, end in HAND_CONNECTIONS:

                x1 = int(hand[start].x * w)
                y1 = int(hand[start].y * h)

                x2 = int(hand[end].x * w)
                y2 = int(hand[end].y * h)

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

            # =================================================
            # FEATURES
            # =================================================

            features = []

            for landmark in hand:

                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            features = np.array(
                features,
                dtype=np.float32
            ).reshape(1, -1)

            # =================================================
            # PREDICTION
            # =================================================

            prediction = model.predict(features)[0]

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(features)[0]

                confidence = float(
                    np.max(probabilities)
                ) * 100


        # =================================================
        # RESULT BOX
        # =================================================

        cv2.rectangle(
            img,
            (10, 10),
            (360, 105),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            img,
            f"Sign: {prediction}",
            (25, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.putText(
            img,
            f"Confidence: {confidence:.1f}%",
            (25, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# =====================================================
# CAMERA
# =====================================================

webrtc_streamer(
    key="sign-language-camera",
    video_processor_factory=SignLanguageProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)


st.markdown("---")

st.write("📱 Mobile Camera Supported")
st.write("🤟 A • B • C • D • E")