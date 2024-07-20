import gradio as gr
import cv2
import numpy as np
import librosa
from tensorflow.keras.models import model_from_json
import pickle
import soundfile as sf
import os 


# Load the pre-trained models
PATH1 = "D:\\Darshit\\AI_Ml_Hackathon\\AI-ML Hackathon - PU - 2024\\Speech-Based Emotion Recognition"

# Speech-based emotion recognition model
with open(f'{PATH1}/CNN_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(f"{PATH1}/CNN_model_weights.h5")

# Facial-based emotion recognition model
with open(f'D:\\Darshit\\AI_Ml_Hackathon\\AI-ML Hackathon - PU - 2024\\Facial-Based Emotion Recognition\\CNN_model-Facial.json', 'r') as json_file:
    loaded_model_json = json_file.read()
video_model = model_from_json(loaded_model_json)
video_model.load_weights('D:\\Darshit\\AI_Ml_Hackathon\\AI-ML Hackathon - PU - 2024\\Facial-Based Emotion Recognition\\CNN_model_weights-Facial.h5')

# Load the scaler and encoder
with open(f'{PATH1}/scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open(f'{PATH1}/encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    try:
        zcr_feat = zcr(data, frame_length, hop_length)
        rmse_feat = rmse(data, frame_length, hop_length)
        mfcc_feat = mfcc(data, sr, frame_length, hop_length)
        
        if zcr_feat.shape[0] != rmse_feat.shape[0] or zcr_feat.shape[0] != mfcc_feat.shape[0]:
            raise ValueError("Feature shapes are not consistent")
        
        result = np.hstack((zcr_feat, rmse_feat, mfcc_feat))
        return result
    except Exception as e:
        print(f"Error extracting features: {e}")
        return np.array([])

def get_predict_feat(aud):
    try:
        res = extract_features(aud[:66150])
        if res.size == 0:
            raise ValueError("Extracted features array is empty.")
        
        result = np.array(res)
        print(f"Shape of extracted features: {result.shape}")
        result = np.reshape(result, newshape=(1, 2376))
        i_result = scaler2.transform(result)
        final_result = np.expand_dims(i_result, axis=2)
        return final_result
    except Exception as e:
        print(f"Error getting prediction features: {e}")
        return np.array([])

def prediction(aud):
    try:
        res = get_predict_feat(aud)
        if res.size == 0:
            raise ValueError("Prediction features array is empty.")
        
        predictions = loaded_model.predict(res)
        y_pred = encoder2.inverse_transform(predictions)
        return y_pred[0][0]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Error"

def preprocess_video_frame(frame):
    try:
        frame_resized = cv2.resize(frame, (224, 224))  # Example size
        frame_normalized = frame_resized / 255.0
        return np.expand_dims(frame_normalized, axis=0)
    except Exception as e:
        print(f"Error preprocessing video frame: {e}")
        return np.array([])
    
def process_audio_and_video(audio, video_frame):
    try:
        if not isinstance(audio, np.ndarray):
            audio = np.array(audio)
        if audio.ndim == 1:
            audio = np.expand_dims(audio, axis=1)
        elif audio.ndim > 2:
            raise ValueError("Audio has too many dimensions")

        # Save the audio locally
        audio_path = "saved_audio/audio.wav"
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)
        with open(audio_path, "wb") as f:
            f.write(audio.tobytes())
        
        audio_output = prediction(audio)

        video_preprocessed = preprocess_video_frame(video_frame)
        if video_preprocessed.size == 0:
            raise ValueError("Preprocessed video frame array is empty.")
        video_predictions = video_model_predict(video_preprocessed)
        video_output = np.argmax(video_predictions[0])

        return audio_output, video_output
    except Exception as e:
        print(f"Error processing audio and video: {e}")
        return str(e), ""
    

def create_gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            audio_input = gr.Audio(label="Record Audio")
            video_input = gr.Video(label="Record Video")

        start_button = gr.Button("Start Processing")

        with gr.Row():
            audio_output = gr.Textbox(label="Audio Model Output")
            video_output = gr.Textbox(label="Video Model Output")

        start_button.click(
            process_audio_and_video,
            inputs=[audio_input, video_input],
            outputs=[audio_output, video_output]
        )

    demo.launch(debug=True)

if __name__ == "__main__":
    create_gradio_interface()
