from clean import preprocess
import pickle
import sqlite3
import pandas as pd
import spacy
import re
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime as dt
import numpy as np
import keras
from keras.models import Sequential 
from keras.preprocessing import sequence
from keras.initializers import he_normal
from keras.layers import BatchNormalization, Dense, Dropout, Flatten, LSTM
from keras.layers.embeddings import Embedding
from keras.regularizers import L1L2
from sklearn.preprocessing import LabelBinarizer
import warnings
warnings.filterwarnings("ignore")
def predict(x):
    out1,acc_svc=predict_svm(x)
    out2,acc_lstm=predict_lstm(x)
    if acc_svc>acc_lstm:
        print(out1)
    else:
        print(out2)



def predict_svm(x):
    x=preprocess(x)
    with open('model_svc.pkl', 'rb') as f:
        model = pickle.load(f)
#     print(model.predict[x])
    with open('label_svm.pkl', 'rb') as f:
        encoder= pickle.load(f)
    
  

    
    
    
    
    
    
    pred=model.predict([x])
    
    pred=encoder.inverse_transform(pred)
    print("output::",pred)
        
    return pred,max(model.predict_proba([x])[0])




def find_word_index(row,word_index_dict):  
    holder = []
    for word in row.split():
        if word in word_index_dict:
            holder.append(word_index_dict[word]) 
        else:
            holder.append(0)            
    return holder
def predict_lstm(x):
    x=preprocess(x)
    with open('model_lstm.pkl', 'rb') as f:
        model = pickle.load(f)
#     print(model.predict[x])
    with open('label_lstm.pkl', 'rb') as f:
        encoder= pickle.load(f)
        
    with open('word_index_dict.pkl', 'rb') as f:
        word_index_dict= pickle.load(f)
        
        
    x= find_word_index(x,word_index_dict)
        
    x=sequence.pad_sequences([x], maxlen=24)
    
    
        
    
    
    
    
    pred_prob=model.predict(x)
    
    print(pred_prob)
    print(encoder.classes_)
    
    pred=encoder.inverse_transform(pred_prob)
    print("output::",pred)
        
    return pred,max(pred_prob[0])



    
    
        
    
    
    
    
    
