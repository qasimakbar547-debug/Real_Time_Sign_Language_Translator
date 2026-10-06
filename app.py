import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Sign Language Translator",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Real-Time Sign Language Translator")
st.write("Show your hand sign to the camera.")

st.info("Supported Signs: A • B • C • D • E")


# =====================================================
# FILE PATHS
# =====================================================

MODEL_PATH = "sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"


# =====================================================
# LOAD RANDOM FOREST MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =====================================================
# MEDIAPIPE
# =====================================================

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions


# =====================================================
# HAND CONNECTIONS
# =====================================================

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

        # Process only 1 out of every 5 frames
        self.frame_count = 0

        # Previous result
        self.prediction = "No Hand"
        self.confidence = 0.0

        # Store landmarks from previous detection
        self.last_hand = None


    def recv(self, frame):

        # =================================================
        # CAMERA FRAME
        # =================================================

        img = frame.to_ndarray(format="bgr24")

        self.frame_count += 1


        # =================================================
        # SMALL RESOLUTION
        # =================================================

        img = cv2.resize(
            img,
            (480, 360),
            interpolation=cv2.INTER_AREA
        )


        # =================================================
        # PROCESS EVERY 5TH FRAME
        # =================================================

        if self.frame_count % 5 == 0:

            rgb = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            self.timestamp += 100

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # =================================================
            # DETECT HAND
            # =================================================

            result = self.landmarker.detect_for_video(
                mp_image,
                self.timestamp
            )


            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                self.last_hand = hand


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

                self.prediction = str(
                    model.predict(features)[0]
                )


                # =================================================
                # CONFIDENCE
                # =================================================

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        features
                    )[0]

                    self.confidence = (
                        float(np.max(probabilities)) * 100
                    )

            else:

                self.last_hand = None
                self.prediction = "No Hand"
                self.confidence = 0.0


        # =================================================
        # DRAW LAST HAND
        # =================================================

        if self.last_hand:

            h, w, _ = img.shape

            for landmark in self.last_hand:

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                cv2.circle(
                    img,
                    (x, y),
                    3,
                    (0, 255, 0),
                    -1
                )


            for start, end in HAND_CONNECTIONS:

                x1 = int(self.last_hand[start].x * w)
                y1 = int(self.last_hand[start].y * h)

                x2 = int(self.last_hand[end].x * w)
                y2 = int(self.last_hand[end].y * h)

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    1
                )


        # =================================================
        # RESULT BOX
        # =================================================

        cv2.rectangle(
            img,
            (10, 10),
            (320, 85),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            img,
            f"Sign: {self.prediction}",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        cv2.putText(
            img,
            f"Confidence: {self.confidence:.1f}%",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )


        # =================================================
        # RETURN CAMERA FRAME
        # =================================================

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


# =====================================================
# WEB CAMERA
# =====================================================

webrtc_streamer(
    key="sign-language-camera",

    video_processor_factory=SignLanguageProcessor,

    media_stream_constraints={
        "video": {
            "width": {"ideal": 480},
            "height": {"ideal": 360},
            "frameRate": {"ideal": 10}
        },
        "audio": False
    },

    async_processing=False
)


# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.write("📱 Mobile Camera Supported")
st.write("🤟 A • B • C • D • E")
