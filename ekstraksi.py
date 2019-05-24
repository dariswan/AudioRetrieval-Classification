#%%
import time
import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
import sklearn.metrics
from sklearn.metrics import  confusion_matrix  
from sklearn.naive_bayes import GaussianNB

start = time.time()
#%% Method for show visual of audio data
"""def display_mfcc(audio):
    y, _ = librosa.load(audio)
    mfcc = librosa.feature.mfcc(y)
    
    plt.figure(figsize=(10,4))
    librosa.display.specshow(mfcc, x_axis='time', y_axis='mel')
    plt.colorbar()
    plt.title(audio)
    plt.tight_layout()
    plt.show()
#%% Using method display_mfcc
display_mfcc('abel1.wav')"""
#%% Method for Extract Feature of audio
def extract_feature_audio(f):
    y, _ = librosa.load(f)

    mfcc = librosa.feature.mfcc(y)
    mfcc /= np.amax(np.absolute(mfcc))
    
    return np.ndarray.flatten(mfcc)[:300]
#%% Collect data and take feature and label by folder's name
def generate_features_and_labels():
    all_features = []
    all_labels = []
    
    ganjils = ['satu','tiga','lima','tujuh','sembilan']
    for ganjil in ganjils:
        sound_files = glob.glob('dataset/'+ganjil+'/*.wav')
        print('Processing %d sound in %s ...' % (len(sound_files), ganjil))
        for f in sound_files:
            features = extract_feature_audio(f)
            all_features.append(features)
            all_labels.append(ganjil)
    return np.stack(all_features), all_labels
#%% method Normalize value Use SoftMax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
#%% Taking feature and label from generate_features_and_labels()
features, labels = generate_features_and_labels()
#%% normalize all feature value
features = softmax(features)
#%% Shuffle data 
alldata = np.column_stack((features,labels))
np.random.seed(200)
np.random.shuffle(alldata)
#%% split data, taking a train and test data
training_split=0.9
splitidx = int(len(alldata)*training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]
#training data
train_input = train[:,:-256]
train_input = train_input.astype(np.float)
train_labels = train[:,-1:]
#testing data
test_input = test[:,:-256]
test_input = test_input.astype(np.float)
test_labels = test[:,-1:]
#%% Method for classification all data from array test_input and take Accuracy or Evaluating
def predict_all():
    nb = GaussianNB()
    nb.fit(train_input, train_labels)
    pred = nb.predict(test_input) 
    
    print("Result Prediction :",pred)
    print("Real label data :",test_labels)
    print('\nEvaluate Naive Bayes')
    print(confusion_matrix(test_labels, pred))  
    print("\nAkurasi : ",sklearn.metrics.accuracy_score(test_labels,pred))
#%% Classification 20% total data
predict_all()

end = time.time()
#print(end-start)







