import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import av

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase


# =====================================================
# PAGE
# =====================================================

st.set_page_config(
    page_title="A-Z Sign Language Translator",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 A-Z Real-Time Sign Language Translator")

st.write(
    "Show an ASL hand sign in front of the camera."
)

st.info(
    "Supported Signs: A • B • C • D • E • F • G • H • I • J • K • L • M"
)

st.info(
    "N • O • P • Q • R • S • T • U • V • W • X • Y • Z"
)


# =====================================================
# FILES
# =====================================================

MODEL_PATH = "sign_model.pkl"
HAND_MODEL_PATH = "hand_landmarker.task"


# =====================================================
# LOAD TRAINED MODEL
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

        self.landmarker = (
            vision.HandLandmarker.create_from_options(
                options
            )
        )

        self.timestamp = 0


    # =================================================
    # PROCESS CAMERA FRAME
    # =================================================

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )


        # ---------------------------------------------
        # Convert BGR to RGB
        # ---------------------------------------------

        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )


        # ---------------------------------------------
        # Timestamp
        # ---------------------------------------------

        self.timestamp += 33


        # ---------------------------------------------
        # MediaPipe Image
        # ---------------------------------------------

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb
        )


        # ---------------------------------------------
        # Detect Hand
        # ---------------------------------------------

        result = self.landmarker.detect_for_video(

            mp_image,

            self.timestamp
        )


        prediction = "No Hand"

        confidence = 0.0


        # =================================================
        # HAND FOUND
        # =================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            h, w, _ = img.shape


            # ---------------------------------------------
            # Draw landmarks
            # ---------------------------------------------

            for landmark in hand:

                x = int(
                    landmark.x * w
                )

                y = int(
                    landmark.y * h
                )

                cv2.circle(

                    img,

                    (x, y),

                    5,

                    (0, 255, 0),

                    -1
                )


            # ---------------------------------------------
            # Draw connections
            # ---------------------------------------------

            for start, end in HAND_CONNECTIONS:

                x1 = int(
                    hand[start].x * w
                )

                y1 = int(
                    hand[start].y * h
                )

                x2 = int(
                    hand[end].x * w
                )

                y2 = int(
                    hand[end].y * h
                )

                cv2.line(

                    img,

                    (x1, y1),

                    (x2, y2),

                    (0, 255, 0),

                    2
                )


            # =================================================
            # CREATE FEATURES
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
            # PREDICT A-Z
            # =================================================

            prediction = str(

                model.predict(features)[0]
            )


            # =================================================
            # CONFIDENCE
            # =================================================

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


        # =================================================
        # RESULT BOX
        # =================================================

        cv2.rectangle(

            img,

            (10, 10),

            (390, 115),

            (0, 0, 0),

            -1
        )


        # ---------------------------------------------
        # SIGN
        # ---------------------------------------------

        cv2.putText(

            img,

            f"Sign: {prediction}",

            (25, 55),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (255, 255, 255),

            2
        )


        # ---------------------------------------------
        # Confidence
        # ---------------------------------------------

        cv2.putText(

            img,

            f"Confidence: {confidence:.1f}%",

            (25, 90),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.6,

            (255, 255, 255),

            2
        )


        # =================================================
        # RETURN FRAME
        # =================================================

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

        "video": {

            "width": {
                "ideal": 480
            },

            "height": {
                "ideal": 360
            },

            "frameRate": {
                "ideal": 10
            }
        },

        "audio": False
    },

    async_processing=True
)


# =====================================================
# SUPPORTED LETTERS
# =====================================================

st.markdown("---")

st.subheader("🤟 Supported ASL Letters")

st.write(

    "A • B • C • D • E • F • G • H • I • J • K • L • M • "
    "N • O • P • Q • R • S • T • U • V • W • X • Y • Z"
)

st.write(
    "📱 Mobile Camera Supported"
)
