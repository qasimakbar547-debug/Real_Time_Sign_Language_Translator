import streamlit as st
import numpy as np
import joblib
import av
import time

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
    "Supported Signs: "
    "A • B • C • D • E • F • G • H • I • J • K • L • M • "
    "N • O • P • Q • R • S • T • U • V • W • X • Y • Z"
)


# =========================================================
# FILE PATHS
# =========================================================

MODEL_PATH = "sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"


# =========================================================
# LOAD TRAINED A-Z MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()


# =========================================================
# MEDIAPIPE SETUP
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

    # IMPORTANT:
    # Python constructor must be _init_
    # NOT init

    def _init_(self):

        # -------------------------------------------------
        # Timestamp
        # -------------------------------------------------

        self.timestamp = 0

        # -------------------------------------------------
        # Create MediaPipe Hand Landmarker
        # -------------------------------------------------

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
    # PROCESS CAMERA FRAME
    # =====================================================

    def recv(self, frame):

        try:

            # -------------------------------------------------
            # Get camera frame
            # -------------------------------------------------

            bgr = frame.to_ndarray(
                format="bgr24"
            )

            # BGR -> RGB
            rgb = bgr[:, :, ::-1].copy()

            height, width, _ = rgb.shape


            # -------------------------------------------------
            # Create MediaPipe Image
            # -------------------------------------------------

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )


            # -------------------------------------------------
            # Generate proper increasing timestamp
            # -------------------------------------------------

            current_timestamp = int(
                time.monotonic() * 1000
            )

            # MediaPipe VIDEO mode requires timestamps
            # to continuously increase.

            if current_timestamp <= self.timestamp:
                current_timestamp = self.timestamp + 1

            self.timestamp = current_timestamp


            # -------------------------------------------------
            # Detect hand
            # -------------------------------------------------

            result = self.landmarker.detect_for_video(
                mp_image,
                self.timestamp
            )


            # -------------------------------------------------
            # Default result
            # -------------------------------------------------

            prediction = "No Hand"
            confidence = 0.0


            # -------------------------------------------------
            # PIL image for drawing
            # -------------------------------------------------

            image = Image.fromarray(rgb)
            draw = ImageDraw.Draw(image)


            # =================================================
            # HAND DETECTED
            # =================================================

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]


                # -------------------------------------------------
                # Draw landmarks
                # -------------------------------------------------

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


                # -------------------------------------------------
                # Draw connections
                # -------------------------------------------------

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


                # -------------------------------------------------
                # Create same 63 features used during training
                # -------------------------------------------------

                features = []

                for landmark in hand:

                    features.extend(
                        [
                            landmark.x,
                            landmark.y,
                            landmark.z
                        ]
                    )


                features = np.array(
                    features,
                    dtype=np.float32
                ).reshape(1, -1)


                # -------------------------------------------------
                # Predict A-Z sign
                # -------------------------------------------------

                prediction = str(
                    model.predict(features)[0]
                )


                # -------------------------------------------------
                # Prediction confidence
                # -------------------------------------------------

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
                            np.max(probabilities)
                        ) * 100
                    )


                # -------------------------------------------------
                # Hand detected message
                # -------------------------------------------------

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
                (10, 10, 430, 115),
                fill=(0, 0, 0)
            )


            # -------------------------------------------------
            # SIGN
            # -------------------------------------------------

            draw.text(
                (25, 30),
                f"Sign: {prediction}",
                fill=(255, 255, 255)
            )


            # -------------------------------------------------
            # CONFIDENCE
            # -------------------------------------------------

            draw.text(
                (25, 70),
                f"Confidence: {confidence:.1f}%",
                fill=(255, 255, 255)
            )


            # -------------------------------------------------
            # RGB -> BGR
            # -------------------------------------------------

            output_rgb = np.array(image)

            output_bgr = (
                output_rgb[:, :, ::-1]
                .copy()
            )


            # -------------------------------------------------
            # Return processed video frame
            # -------------------------------------------------

            return av.VideoFrame.from_ndarray(
                output_bgr,
                format="bgr24"
            )


        # =====================================================
        # ERROR HANDLING
        # =====================================================

        except Exception as e:

            bgr = frame.to_ndarray(
                format="bgr24"
            )

            rgb = bgr[:, :, ::-1].copy()

            image = Image.fromarray(rgb)

            draw = ImageDraw.Draw(image)


            # -------------------------------------------------
            # Error box
            # -------------------------------------------------

            draw.rectangle(
                (10, 10, 700, 100),
                fill=(0, 0, 0)
            )


            draw.text(
                (20, 25),
                "PROCESSING ERROR",
                fill=(255, 0, 0)
            )


            draw.text(
                (20, 55),
                str(e)[:100],
                fill=(255, 255, 255)
            )


            output_rgb = np.array(image)

            output_bgr = (
                output_rgb[:, :, ::-1]
                .copy()
            )


            return av.VideoFrame.from_ndarray(
                output_bgr,
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
