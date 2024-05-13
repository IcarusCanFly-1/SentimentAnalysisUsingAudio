
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
import matplotlib.pyplot as plt
# Any results you write to the current directory are saved as output.
###### import libraries
import librosa
import wave
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
def extract_mfcc(wav_file_name):

    y, sr = librosa.load(wav_file_name)
#     trimmed_data = np.zeros((160, 20))
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
#     data = np.array(librosa.feature.mfcc(y = y, sr = sr, n_mfcc=40).T)
#     if data.shape[0] <= 160:
#         trimmed_data[:data.shape[0],0:] = data[:,0:]
#     else:
#         trimmed_data[0:,0:] = data[0:160,0:]
    return mfccs
### extract audio data from AV RAVDESS data
# root_dir = r"D:\audio_emotion_classification\RAVDESS_files\03-01-03-01-01-02-01.wav"
#
# audio_only_data = [] ###stores the mfcc data
# audio_only_labels = [] ###stores the labels
# for subdirs, dirs, files in os.walk(root_dir):
#     for file in files:
#         y, sr = librosa.load(os.path.join(subdirs,file))
#         mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T,axis=0)
#         audio_only_data.append(mfccs)
#         audio_only_labels.append(int(file[7:8]) - 1)
# #### convert data to array and make labels categorical
# audio_only_data_array = np.array(audio_only_data)
# audio_only_labels_array = np.array(audio_only_labels)
# audio_only_data_array.shape
##### load data from savee dataset
#### although, we load the data here, it is not used in training or validation
# root_dir = "../input/savee-emotion-recognition/audiodata/AudioData/"
# # root_dir = "../input/audio_speech_actors_01-24/"
# savee_data = []
# savee_labels = []
# # Update root_dir to point to the 'ALL' folder inside the 'SAVEE' directory
# root_dir = os.path.join(root_dir, 'SAVEE', 'ALL')
#
# for actor_dir in sorted(os.listdir(root_dir)):
#     if actor_dir[-4:] == ".txt":
#         continue
#     for file_name in os.listdir(os.path.join(root_dir, actor_dir)):
#         if file_name[0] == "c":
#             continue
#         wav_file_name = os.path.join(root_dir, actor_dir, file_name)
#         savee_data.append(extract_mfcc(wav_file_name))
#         if file_name[0] == "n":
#             savee_labels.append(0)
#         if file_name[0] == "a":
#             savee_labels.append(4)
#         if file_name[0] == "d":
#             savee_labels.append(6)
#         if file_name[0] == "f":
#             savee_labels.append(5)
#         if file_name[0] == "h":
#             savee_labels.append(2)
#         if file_name[:2] == "sa":
#             savee_labels.append(3)
#         if file_name[:2] == "su":
#             savee_labels.append(7)
# #### convert data to array and make labels categorical
# savee_data_array = np.asarray(savee_data)
# savee_label_array = np.array(savee_labels)
# to_categorical(savee_label_array)[0].shape
# # savee_data_array.shape
##### load radvess speech data #####
root_dir = r"D:\audio_emotion_classification\Audio_Speech_Actors_01-24"
# root_dir = "../input/audio_speech_actors_01-24/"
# actor_dir = os.listdir("../input/audio_speech_actors_01-24/")
radvess_speech_labels = []
ravdess_speech_data = []
for actor_dir in sorted(os.listdir(root_dir)):
    actor_name = os.path.join(root_dir, actor_dir)
    for file in os.listdir(actor_name):
        radvess_speech_labels.append(int(file[7:8]) - 1)
        wav_file_name = os.path.join(root_dir, actor_dir, file)
        ravdess_speech_data.append(extract_mfcc(wav_file_name))
#### convert data to array and make labels categorical
ravdess_speech_data_array = np.asarray(ravdess_speech_data)
ravdess_speech_label_array = np.array(radvess_speech_labels)
ravdess_speech_label_array.shape
### load RAVDESS song data
root_dir = r"D:\audio_emotion_classification\Audio_Song_Actors_01-24"
radvess_song_labels = []
ravdess_song_data = []
for actor_dir in sorted(os.listdir(root_dir)):
    actor_name = os.path.join(root_dir, actor_dir)
    for file in os.listdir(actor_name):
        radvess_song_labels.append(int(file[7:8]) - 1)
        wav_file_name = os.path.join(root_dir, actor_dir, file)
        ravdess_song_data.append(extract_mfcc(wav_file_name))
#### convert data to array and make labels categorical
ravdess_song_data_array = np.asarray(ravdess_song_data)
ravdess_song_label_array = np.array(radvess_song_labels)
ravdess_song_label_array.shape
# #### combine data


data = np.r_[ravdess_speech_data_array, ravdess_song_data_array]
labels = np.r_[ravdess_speech_label_array, ravdess_song_label_array]
# data = ravdess_speech_data_array
# labels = ravdess_speech_label_array
labels.shape
### plot a histogram to understand the distribution of the data
import matplotlib.pyplot as plt
plt.hist(labels)
plt.show()
### make categorical labels
labels_categorical = to_categorical(labels)
data.shape
labels_categorical.shape
def create_model_LSTM():
    ### LSTM model, referred to the model A in the report
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

number_of_samples = data.shape[0]
training_samples = int(number_of_samples * 0.8)
validation_samples = int(number_of_samples * 0.1)
test_samples = int(number_of_samples * 0.1)
### train using model A
model_A = create_model_LSTM()
history = model_A.fit(np.expand_dims(data[:training_samples],-1), labels_categorical[:training_samples], validation_data=(np.expand_dims(data[training_samples:training_samples+validation_samples], -1), labels_categorical[training_samples:training_samples+validation_samples]), epochs=100, shuffle=True)
### loss plots using model A
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
### accuracy plots using model A
plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
### evaluate using model A
model_A.evaluate(np.expand_dims(data[training_samples + validation_samples:], -1), labels_categorical[training_samples + validation_samples:])
# model.evaluate(predictions, labels_categorical[training_samples + validation_samples:])
import seaborn as sn
from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(model_A.predict_classes(np.expand_dims(data[training_samples + validation_samples:], -1)), labels[training_samples + validation_samples:])
# sn.set(font_scale=1.4)#for label size
# sn.heatmap(cm, annot=True,annot_kws={"size": 16})# font size
# Predict probabilities for each class
predictions_prob = model_A.predict(np.expand_dims(data[training_samples + validation_samples:], -1))
# Get the class with the highest probability for each prediction
predicted_classes = np.argmax(predictions_prob, axis=1)

# Now, you can calculate the confusion matrix using the predicted classes
cm = confusion_matrix(predicted_classes, labels[training_samples + validation_samples:])

model_A.save_weights("Model_A.weights.h5")




