import cv2
import streamlit as st
import os
import replicate
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase, VideoProcessorBase

# Set your Replicate API token
replicate_api = 'r8_XPfmMWkjS7hMxocjjt9jdjlW4Ju4pS30rbLVe'
os.environ['REPLICATE_API_TOKEN'] = replicate_api

client = replicate.Client(api_token=os.getenv('REPLICATE_API_TOKEN'))

# Load the pre-trained Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to generate technical questions
def generate_technical(description):
    prompt = f"You are a knowledgeable assistant specialized in technology. Based on the following description, generate 2 technical questions : {description}"
    output = replicate.run('meta/meta-llama-3-70b-instruct',
                           input={"prompt": prompt,
                                  "temperature": 0.1, "top_p": 0.9, "max_length": 512, "repetition_penalty": 1})
    return "".join(output)

# Video processor class for face detection
class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_cascade = face_cascade

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert the image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        return img

def main():
    st.title("Real-time Video Processing with Streamlit WebRTC")

    # Define RTC Configuration for STUN/TURN servers (optional)
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    # Define a WebRTC video streamer component
    webrtc_ctx = webrtc_streamer(
        key="example",
        rtc_configuration=rtc_configuration,
        video_processor_factory=FaceDetectionProcessor,
        media_stream_constraints={"video": True, "audio": False},
    )

    # Example usage of the generate_technical function
    description = st.text_input("Enter a description to generate technical questions:")
    if st.button("Generate"):
        questions = generate_technical(description)
        st.write(questions)

if __name__ == "__main__":
    main()
