import speech_recognition as sr
import pyttsx3

# initialize the recognizer and the text-to-speech engine
r = sr.Recognizer()
engine = pyttsx3.init()

# use the microphone as source
with sr.Microphone() as source:
    # adjust for ambient noise
    r.adjust_for_ambient_noise(source)
    
    # listen for 5 seconds to capture speech
    print("Say something...")
    audio = r.listen(source, timeout=5)
    
# recognize speech using Google Speech Recognition
try:
    text = r.recognize_google(audio)
    print("The Text is  " + text)
        
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
# except sr.RequestError as e:
#     print("Could not request results from Google Speech Recognition service; {0}".format(e))