from clean import preprocess
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
from sklearn.model_selection import GridSearchCV
import pickle
import warnings
warnings.filterwarnings("ignore")
class twitter():


    def training(self,x,y):
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=35)
        param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001]} 
        pl = Pipeline([('tfidf',TfidfVectorizer()),('clf',GridSearchCV(SVC(probability=True), param_grid, refit = True, verbose = 3))])
        pl.fit(x_train,y_train)
        predicts = pl.predict(x_test)
        print(confusion_matrix(y_test,predicts))
        print(classification_report(y_test,predicts))
        print("accuracy::",accuracy_score(y_test,predicts))
        with open('model_svc.pkl','wb') as f:
            pickle.dump(pl,f)
        return pl,accuracy_score(y_test,predicts)


    def data_gen(self,data):
        y=data['airline_sentiment']
        le = LabelEncoder()
        y=le.fit_transform(y)
        pickle.dump(le,open('label_svm.pkl', 'wb'))
        data.text = data.text.apply(lambda x: preprocess(x))
        print("number of classes::",len(list(le.classes_)))
        nclasses=len(list(le.classes_))
        x=data.text
        return x,y,nclasses

   




   


    def find_word_index(self,row,word_index_dict):  
        holder = []
        for word in row.split():
            if word in word_index_dict:
                holder.append(word_index_dict[word]) 
            else:
                holder.append(0)            
        return holder



    def lstm_training(self,x,y):
        total_words = []

        for sent in x:
            words = sent.split()
            total_words+=words
        from collections import Counter
        counter = Counter(total_words)
        top_words_count = int(len(counter)/0.95)
        sorted_words = counter.most_common(top_words_count)

        word_index_dict = dict()
        i = 1
        for word,frequency in sorted_words:
            word_index_dict[word] = i
            i += 1
        text=[]
        for t in x:
            text.append(self.find_word_index(t,word_index_dict))
        pickle.dump(le,open('word_index_dict.pkl', 'wb'))
        label_binarizer = LabelBinarizer()
        labels = label_binarizer.fit_transform(y)
        pickle.dump(label_binarizer,open('label_lstm.pkl', 'wb'))
        n_classes = len(label_binarizer.classes_)
        x_train,x_test,y_train,y_test = train_test_split(text,labels,test_size=0.1,shuffle=True,random_state=35)
        m=0
        for ind in text:
            i=len(ind)
            m=max(m,i)
        max_review_length = m

        x_train = sequence.pad_sequences(x_train, maxlen=max_review_length)
        x_test = sequence.pad_sequences(x_test, maxlen=max_review_length)
        print("nclasses:",n_classes)
        print("max length:",m)
        vocab_size = len(counter.most_common()) + 1
        model = Sequential()

    # Add Embedding Layer
        model.add(Embedding(vocab_size, 32, input_length=max_review_length))

    # Add batch normalization
        model.add(BatchNormalization())

    # Add dropout
        model.add(Dropout(0.20))

    # Add LSTM Layer
        model.add(LSTM(128,return_sequences=True))

        model.add(LSTM(64))

    # Add dropout
        model.add(Dropout(0.20))

    # Add Dense Layer
        model.add(Dense(3, activation='softmax'))

    # Summary of the model
        print("Model Summary: \n")
        print(model.summary())



        callbacks = [ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1,restore_best_weights=True)
        
        results = model.fit(x_train, np.array(y_train), batch_size = 32, epochs = 10, verbose=2, validation_data=(x_test, y_test),callbacks=[early_stop])
        test_scores = model.evaluate(x_test,y_test,verbose=1)
        accuracy=test_scores[1]
        predicts = model.predict(x_test)
        
    #     print("accuracy::",accuracy_score(y_test,predicts))
        with open('model_lstm.pkl','wb') as f:
            pickle.dump(model,f)
        
        return model,accuracy



def main():
    data = pd.read_csv("Tweets.csv")
    print("done")

    t=twitter()
    
    x,y,nclasses=t.data_gen(data)
    print("done")
    svm_model,accuracy_svm=t.training(x,y)
    print("done")
    lstm_model,accuracy_lstm=t.lstm_training(x,y)
    
    if(accuracy_lstm>=accuracy_svm):
        with open('best_model/model.pkl','wb') as f:
            pickle.dump(lstm_model,f)
            print("LSTM")
    else:
        with open('best_model/model.pkl','wb') as f:
            pickle.dump(svm_model,f)
            print("SVM")
        
    
    

        
    
        

if __name__ == "__main__": 
    # calling main function 
    main() 
        
    
        
