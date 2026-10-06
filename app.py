import streamlit as st
import numpy as np
import joblib
import av
import time
import threading

from PIL import Image, ImageDraw
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="A-Z Sign Language Translator",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 A-Z Real-Time Sign Language Translator")

st.write(
    "Show your ASL hand sign clearly in front of the camera."
)

st.info(
    "Supported Signs: A • B • C • D • E • F • G • H • I • J • K • L • M • "
    "N • O • P • Q • R • S • T • U • V • W • X • Y • Z"
)


# =========================================================
# FILES
# =========================================================

MODEL_PATH = "sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# MEDIAPIPE
# =========================================================

BaseOptions = python.BaseOptions


# =========================================================
# HAND CONNECTIONS
# =========================================================

HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


# =========================================================
# VIDEO PROCESSOR
# =========================================================

class SignLanguageProcessor(VideoProcessorBase):

    def _init_(self):

        # -----------------------------------------------
        # Call parent constructor
        # -----------------------------------------------

        try:
            super()._init_()
        except Exception:
            pass

        # -----------------------------------------------
        # Safe initial state
        # -----------------------------------------------

        self.timestamp = 0
        self.landmarker = None
        self.lock = threading.Lock()

        # -----------------------------------------------
        # Create MediaPipe
        # -----------------------------------------------

        self.create_landmarker()


    # =====================================================
    # CREATE LANDMARKER
    # =====================================================

    def create_landmarker(self):

        options = vision.HandLandmarkerOptions(

            base_options=BaseOptions(
                model_asset_path=HAND_MODEL_PATH
            ),

            running_mode=vision.RunningMode.VIDEO,

            num_hands=1,

            min_hand_detection_confidence=0.3,

            min_hand_presence_confidence=0.3,

            min_tracking_confidence=0.3
        )

        self.landmarker = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )


    # =====================================================
    # GET SAFE TIMESTAMP
    # =====================================================

    def get_timestamp(self):

        # IMPORTANT:
        # Never assume timestamp already exists.

        old_timestamp = getattr(
            self,
            "timestamp",
            0
        )

        current_timestamp = int(
            time.monotonic() * 1000
        )

        if current_timestamp <= old_timestamp:

            current_timestamp = (
                old_timestamp + 1
            )

        self.timestamp = current_timestamp

        return current_timestamp


    # =====================================================
    # PROCESS FRAME
    # =====================================================

    def recv(self, frame):

        try:

            # -------------------------------------------
            # Make sure state exists
            # -------------------------------------------

            if not hasattr(
                self,
                "timestamp"
            ):
                self.timestamp = 0

            if not hasattr(
                self,
                "landmarker"
            ) or self.landmarker is None:

                self.create_landmarker()


            # -------------------------------------------
            # Camera frame
            # -------------------------------------------

            bgr = frame.to_ndarray(
                format="bgr24"
            )

            rgb = bgr[:, :, ::-1].copy()

            height, width, _ = rgb.shape


            # -------------------------------------------
            # MediaPipe image
            # -------------------------------------------

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )


            # -------------------------------------------
            # SAFE TIMESTAMP
            # -------------------------------------------

            timestamp = self.get_timestamp()


            # -------------------------------------------
            # MediaPipe detection
            # -------------------------------------------

            result = self.landmarker.detect_for_video(
                mp_image,
                timestamp
            )


            # -------------------------------------------
            # Default
            # -------------------------------------------

            prediction = "No Hand"
            confidence = 0.0


            # -------------------------------------------
            # PIL image
            # -------------------------------------------

            image = Image.fromarray(rgb)

            draw = ImageDraw.Draw(image)


            # =================================================
            # HAND FOUND
            # =================================================

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]


                # -------------------------------------------
                # Draw landmarks
                # -------------------------------------------

                for landmark in hand:

                    x = int(
                        landmark.x * width
                    )

                    y = int(
                        landmark.y * height
                    )

                    draw.ellipse(
                        (
                            x - 5,
                            y - 5,
                            x + 5,
                            y + 5
                        ),
                        fill=(0, 255, 0)
                    )


                # -------------------------------------------
                # Draw connections
                # -------------------------------------------

                for start, end in HAND_CONNECTIONS:

                    x1 = int(
                        hand[start].x * width
                    )

                    y1 = int(
                        hand[start].y * height
                    )

                    x2 = int(
                        hand[end].x * width
                    )

                    y2 = int(
                        hand[end].y * height
                    )

                    draw.line(
                        (
                            x1,
                            y1,
                            x2,
                            y2
                        ),
                        fill=(0, 255, 0),
                        width=3
                    )


                # -------------------------------------------
                # CREATE 63 FEATURES
                # -------------------------------------------

                features = []

                for landmark in hand:

                    features.append(
                        landmark.x
                    )

                    features.append(
                        landmark.y
                    )

                    features.append(
                        landmark.z
                    )


                features = np.array(
                    features,
                    dtype=np.float32
                ).reshape(1, 63)


                # -------------------------------------------
                # MODEL PREDICTION
                # -------------------------------------------

                prediction = str(
                    model.predict(
                        features
                    )[0]
                )


                # -------------------------------------------
                # CONFIDENCE
                # -------------------------------------------

                if hasattr(
                    model,
                    "predict_proba"
                ):

                    probabilities = (
                        model.predict_proba(
                            features
                        )[0]
                    )

                    confidence = (
                        float(
                            np.max(
                                probabilities
                            )
                        ) * 100
                    )


                # -------------------------------------------
                # HAND DETECTED
                # -------------------------------------------

                draw.text(
                    (20, 135),
                    "HAND DETECTED",
                    fill=(0, 255, 0)
                )


            # =================================================
            # NO HAND
            # =================================================

            else:

                draw.text(
                    (20, 135),
                    "SHOW YOUR HAND",
                    fill=(255, 0, 0)
                )


            # =================================================
            # RESULT BOX
            # =================================================

            draw.rectangle(
                (10, 10, 440, 115),
                fill=(0, 0, 0)
            )


            # -------------------------------------------
            # SIGN
            # -------------------------------------------

            draw.text(
                (25, 30),
                f"Sign: {prediction}",
                fill=(255, 255, 255)
            )


            # -------------------------------------------
            # CONFIDENCE
            # -------------------------------------------

            draw.text(
                (25, 70),
                f"Confidence: {confidence:.1f}%",
                fill=(255, 255, 255)
            )


            # -------------------------------------------
            # RGB -> BGR
            # -------------------------------------------

            output = np.array(image)

            output = (
                output[:, :, ::-1]
                .copy()
            )


            # -------------------------------------------
            # RETURN FRAME
            # -------------------------------------------

            return av.VideoFrame.from_ndarray(
                output,
                format="bgr24"
            )


        # =================================================
        # ERROR
        # =================================================

        except Exception as e:

            # -------------------------------------------
            # Camera frame
            # -------------------------------------------

            bgr = frame.to_ndarray(
                format="bgr24"
            )

            rgb = (
                bgr[:, :, ::-1]
                .copy()
            )

            image = Image.fromarray(rgb)

            draw = ImageDraw.Draw(image)


            # -------------------------------------------
            # Error box
            # -------------------------------------------

            draw.rectangle(
                (10, 10, 720, 105),
                fill=(0, 0, 0)
            )


            draw.text(
                (20, 25),
                "PROCESSING ERROR",
                fill=(255, 0, 0)
            )


            draw.text(
                (20, 55),
                str(e)[:110],
                fill=(255, 255, 255)
            )


            # -------------------------------------------
            # Return frame
            # -------------------------------------------

            output = np.array(image)

            output = (
                output[:, :, ::-1]
                .copy()
            )

            return av.VideoFrame.from_ndarray(
                output,
                format="bgr24"
            )


# =========================================================
# CAMERA
# =========================================================

webrtc_streamer(

    key="sign-language-camera",

    video_processor_factory=SignLanguageProcessor,

    media_stream_constraints={
        "video": {
            "width": {
                "ideal": 640
            },

            "height": {
                "ideal": 480
            },

            "frameRate": {
                "ideal": 10
            }
        },

        "audio": False
    },

    async_processing=True
)


# =========================================================
# SUPPORTED LETTERS
# =========================================================

st.markdown("---")

st.subheader(
    "🤟 Supported ASL Letters"
)

st.write(
    "A • B • C • D • E • F • G • H • I • J • K • L • M • "
    "N • O • P • Q • R • S • T • U • V • W • X • Y • Z"
)

st.write(
    "📱 Mobile Camera Supported"
)
