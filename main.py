import tkinter
from tkinter import *
from tkinter.filedialog import askopenfilename
from playsound import playsound
import numpy as np
from keras.models import Sequential
from keras.layers import *
import keras
import librosa
from PIL import Image, ImageTk  # For displaying images

top = Tk()
top.title('Emotion recognition from audio')
top.geometry('950x350')

# Define function to show accuracy graph
def show_accuracy_graph():
    # Load accuracy graph image
    accuracy_image = Image.open(r"D:\audio_emotion_classification\Figure_1.png")
    accuracy_image = accuracy_image.resize((400, 300), resample=Image.BILINEAR)
    accuracy_photo = ImageTk.PhotoImage(accuracy_image)

    # Display accuracy graph
    accuracy_label = Label(top, image=accuracy_photo)
    accuracy_label.image = accuracy_photo
    accuracy_label.grid(row=6, column=0, columnspan=2)

# Button to show accuracy graph
accuracy_button = Button(top, text='Show accuracy graph', command=show_accuracy_graph)
accuracy_button.grid(row=5, column=0, columnspan=2)

# canvas.grid(row=1, column=0)

def openAudio():
    ''' Opens the audio file'''
    File = askopenfilename(title='Open an Audio file')
    e.set(File)


def playAudio():
    ''' Play the audio file '''
    playsound(e.get())


e = StringVar()
submit_button = Button(top, text='Open an Audio file', command=openAudio)
submit_button.grid(row=1, column=0)

submit_button = Button(top, text='Play an Audio File', command=playAudio)
submit_button.grid(row=3, column=0)

emotions_used = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def model_A():
    ''' LSTM model definition and architecture
    The model returned here is referred to as model A
    '''
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
    ''' Prediction using model A'''
    mfcc = extract_mfcc(e.get())
    model = model_A()
    model.load_weights(r"C:\Users\Suchismita\Downloads\audio\Audio_emotion_recognition-master\Model_A.h5")
    mfcc = np.expand_dims(mfcc, -1)
    mfcc = np.expand_dims(mfcc, 0)
    print(mfcc.shape)
    cls_wav = model.predict(mfcc)

    # Find the index of the maximum probability in cls_wav
    max_index = np.argmax(cls_wav)

    # Get the corresponding emotion from emotions_used
    predicted_emotion = emotions_used[max_index]

    # Prepare the text to be displayed
    textvar = "The predicted emotion is: %s" % predicted_emotion

    # Clear the text widget
    t1.delete(0.0, tkinter.END)

    # Insert the text into the text widget
    t1.insert('insert', textvar + '\n')

    # Update the text widget
    t1.update()


def extract_mfcc(wav_file_name):
    ''' Extracts mfcc features and outputs the average of each dimension'''
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

l1 = Label(top,
           text='First character in the SAVEE data files represent the labels, eg n01.wav has label neutral, where')
l1.grid(row=12)

l1 = Label(top, text='d=disgust, f=fearful, sa=sadness, su=surprised')
l1.grid(row=13)

t1 = Text(top, bd=0, width=20, height=2, font='Fixdsys -14')
t1.grid(row=0, column=1)

top.mainloop()
