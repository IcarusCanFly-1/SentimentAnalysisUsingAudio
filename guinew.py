import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.messagebox import showinfo
import sounddevice as sd
import soundfile as sf
import numpy as np
from keras.models import Sequential
from keras.layers import *
import librosa
from PIL import Image, ImageTk
import os

def resize_image(event):
    global bg_image, bg_photo
    size = (event.width, event.height)
    resized_image = bg_image.resize(size, Image.BILINEAR)
    bg_photo = ImageTk.PhotoImage(resized_image)
    bg_label.config(image=bg_photo)

top = Tk()
top.title('Emotion recognition from audio')
top.geometry('950x350')

# Load background image
bg_image = Image.open(r"D:\downloads\emotion_classification_Audio\v627-aew-21-technologybackground.jpg")  # Insert the path to your background image
bg_photo = ImageTk.PhotoImage(bg_image)

# Create a label to hold the background image
bg_label = Label(top, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Set label to fill the entire window

# Bind the resize event of the window to the resize_image function
top.bind("<Configure>", resize_image)

# Define function to show training and validation loss graph
def show_loss_graph():
    # Load loss graph image
    loss_image = Image.open(r"D:\emotion_classification_Audio\emotion_classification_Audio\Figure_1.png")
    loss_image = loss_image.resize((400, 300), resample=Image.BILINEAR)
    loss_photo = ImageTk.PhotoImage(loss_image)

    # Display loss graph
    loss_label = Label(top, image=loss_photo)
    loss_label.image = loss_photo
    loss_label.grid(row=5, column=1, columnspan=2, pady=(10, 0), rowspan=2)

loss_button = Button(top, text='Training and Validation Loss', command=show_loss_graph)
loss_button.grid(row=3, column=0, columnspan=2, sticky=E)

# Function to record audio and save to a file
def record_audio(file_path, duration=5, fs=44100):
    print("Recording...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    sf.write(file_path, audio_data, fs)
    print("Recording finished.")

# Function to open file dialog for selecting audio file
def openAudio():
    File = askopenfilename(title='Open an Audio file')
    e.set(File)

# Function to play selected audio file
def playAudio():
    showinfo("Play Audio", "Playing audio. Click OK to start.")
    sd.play(sf.read(e.get())[0], samplerate=44100)
    sd.wait()

# Function to record audio and save it in the same directory
def record_and_save():
    file_path = os.path.join(os.getcwd(), "recorded_audio.wav")
    record_audio(file_path)
    showinfo("Recording Finished", "Audio recorded and saved successfully at: {}".format(file_path))

e = StringVar()
submit_button = Button(top, text='Open an Audio file', command=openAudio)
submit_button.grid(row=1, column=0)

submit_button = Button(top, text='Play an Audio File', command=playAudio)
submit_button.grid(row=2, column=0)

record_button = Button(top, text='Record Audio and Save', command=record_and_save)
record_button.grid(row=2, column=1)

# Define emotions used globally
emotions_used = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def model_A():
    model = Sequential()
    model.add(LSTM(128, return_sequences=False, input_shape=(40, 1)))
    model.add(Dense(64))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(32))
    model.add(Dropout(0.4))
    model.add(Activation('relu'))
    model.add(Dense(8))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return model

def Predict_A():
    file_path = e.get()
    mfcc = extract_mfcc(file_path)
    model = model_A()
    model.load_weights(r"D:\emotion_classification_Audio\emotion_classification_Audio\Model_A.h5")
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    print(mfcc.shape)
    cls_wav = model.predict(mfcc)

    max_index = np.argmax(cls_wav)
    predicted_emotion = emotions_used[max_index]

    textvar = "The predicted emotion is: %s" % predicted_emotion
    t1.delete(0.0, tkinter.END)
    t1.insert('insert', textvar + '\n')
    t1.update()

def extract_mfcc(wav_file_name):
    y, sr = librosa.load(wav_file_name)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

submit_button = Button(top, text='Predict using Model A', command=Predict_A)
submit_button.grid(row=1, column=1)

l1 = Label(top, text='Press <Open> to open, <Play> to play an audio file, then press <Predict> ')
l1.grid(row=7)

l1 = Label(top, text='')
l1.grid(row=8)

l1 = Label(top,
           text='7-8th letters in the RAVDESS data file names represent the labels, eg 03-01-02-01-01-01-01.wav has label 02, where')
l1.grid(row=9)

l1 = Label(top, text='01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised')
l1.grid(row=10)

l1 = Label(top, text='')
l1.grid(row=11)

t1 = Text(top, bd=0, width=20, height=2, font='Fixdsys -14')
t1.grid(row=0, column=1)

top.mainloop()
