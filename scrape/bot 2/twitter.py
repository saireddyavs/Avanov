import twint

c = twint.Config()

c.Search=input("Enter sentence::")
c.Limit = 100
c.Email=True

c.Store_csv = True
c.Output = "none"
twint.run.Search(c)

import mysql.connector as mysql

mydb = mysql.connect(
  host="127.0.0.1",
  user="sai",
  passwd="rgukt123",
    port="3309"
)

print(mydb)

# import pandas as pd

# df=pd.read_csv("none/tweets.csv")

# print(df.columns)

mycursor = mydb.cursor()



try:
    mycursor.execute("CREATE DATABASE bot")
except:
    print("Database named bot already exists")






mydb = mysql.connect(
  host="localhost",
  user="root",
  passwd="rgukt123",
  database="bot",
    port="3309"
    
)


mycursor = mydb.cursor()

# sql = "DROP TABLE IF EXISTS Twitter"

# mycursor.execute(sql) 

#mycursor.execute("CREATE TABLE Twitter (id VARCHAR(255), conversation_id VARCHAR(255), created_at VARCHAR(255), date VARCHAR(255), time VARCHAR(255), timezone VARCHAR(255),user_id VARCHAR(255) , username VARCHAR(255), name VARCHAR(255), place VARCHAR(255), tweet TEXT, mentions TEXT, urls VARCHAR(255),photos VARCHAR(255), replies_count VARCHAR(255), retweets_count VARCHAR(255), likes_count VARCHAR(255), hashtags VARCHAR(255),cashtags VARCHAR(255), link VARCHAR(255), retweet VARCHAR(255), quote_url VARCHAR(255) , video VARCHAR(255), near VARCHAR(255), geo VARCHAR(255),source VARCHAR(255), user_rt_id VARCHAR(255), user_rt VARCHAR(255),retweet_id VARCHAR(255), reply_to TEXT,retweet_date VARCHAR(255),sentiment text,confidence text)");


import pandas as pd

df=pd.read_csv("none/tweets.csv")

print(df.columns)
# question.isnull().any()
# question.columns
import datetime as dt
import nltk
start = dt.datetime.now()
sno=nltk.stem.SnowballStemmer("english")
i=0
str1=""
final_string=[]
all_positive_words=[]
all_negative_words=[]
s=""
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
excluding = ['against','not','don', "don't",'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't",
             'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 
             'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",'shouldn', "shouldn't", 'wasn',
             "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
stop = [words for words in stop if words not in excluding]

import re
def cleanhtml(sentence):
    cleanr=re.compile('<.*>')
    cleantext=re.sub(cleanr,'',sentence)
    return cleantext
def cleanpunc(sentence):
    cleaned=re.sub(r'[?|!|\'|"|#]',r' ',sentence) #seee \' and combination
    cleaned=re.sub(r'[.|,|)|/|\|(]',r' ',cleaned)
    return cleaned
print(sno.stem("tasty"))#checking the rootword of tasty

print(sno)



def preprocess1(X):
    final_string=[]
    

    for sent in X:
        filterd_sentence=[]
        
        sent=str(sent)

        sent=cleanhtml(sent)
#         print(sent)
        for w in sent.split():
            for cleaned_words in cleanpunc(w).split():
                if((cleaned_words.isalpha())&(len(cleaned_words)>2)):
                    if(cleaned_words.lower() not in stop):
                        s=(sno.stem(cleaned_words.lower())).encode("utf8")
                        filterd_sentence.append(s)

                    else:
                        continue
                else:
                    continue

        str1= b" ".join(filterd_sentence)
        final_string.append(str1)
#         i=i+1
    return final_string


def preprocess(X):
    final_string=[]
    X=[X]

    for sent in X:
        filterd_sentence=[]
        
        sent=str(sent)

        sent=cleanhtml(sent)
#         print(sent)
        for w in sent.split():
            for cleaned_words in cleanpunc(w).split():
                if((cleaned_words.isalpha())&(len(cleaned_words)>2)):
                    if(cleaned_words.lower() not in stop):
                        s=(sno.stem(cleaned_words.lower())).encode("utf8")
                        filterd_sentence.append(s)

                    else:
                        continue
                else:
                    continue

        str1= b" ".join(filterd_sentence)
        final_string.append(str1)
#         i=i+1
    return final_string



import pickle
from keras.preprocessing import sequence
def find_word_index(row,word_index_dict):  
    holder = []
    for word in row.split():
        if word in word_index_dict:
            holder.append(word_index_dict[word]) 
        else:
            holder.append(0)            
    return holder
def predict_lstm(x):
    x=preprocess1(x)
#     print(x)
#     with open('model_lstm.pkl', 'rb') as f:
#         model = pickle.load(f)

#     print(model.predict[x])
    with open('label_transform.pkl', 'rb') as f:
        encoder= pickle.load(f)
        
    from keras.models import load_model
    model = load_model('LSTM_1.ckpt')
        
    with open('word_index_dict.pkl', 'rb') as f:
        word_index_dict= pickle.load(f)
        
    text=[]
    for sent in x:
        text.append(find_word_index(sent,word_index_dict))
        
        
        
  
    #print(len(x),len(text))  
    x=sequence.pad_sequences(text, maxlen=500)
    
    
        
    
    
    
    
    pred_prob=model.predict(x)
    
#     print(pred_prob)
#     print(encoder.classes_)
    
    
    sentiment=[]
    
#     print(encoder.inverse_transform(pred_prob))
    
    preds=encoder.inverse_transform(pred_prob)
    
    for pre in preds:
        if pre==1:
            sentiment.append("Strongly Negative")
        if pre==2:
            sentiment.append("Weekly Negative")
        if pre==3:
            sentiment.append("Neutral")
        if pre==4:
            sentiment.append("Weekly Positive")
        if pre==5:
            sentiment.append("Strongly Positive")
    
  
            
        
    


    
    
#     print("output::",pred)
    
        
    return [sentiment,pred_prob]


df['cleaned']=df.tweet.apply(lambda x:preprocess(x))


# df.head()
q=predict_lstm(list(df.cleaned))
df['sentiment']=q[0]


import numpy as np

df['confidence']=q[1].tolist()



df.drop('cleaned', axis=1, inplace=True)
df = df.astype(str)


for i in df.index:
    if i==0:
        continue
    
    mycursor.execute('INSERT INTO Twitter(id , conversation_id , created_at , date , time , timezone ,user_id  , username , name , place , tweet , mentions , urls ,photos , replies_count , retweets_count , likes_count , hashtags ,cashtags , link , retweet , quote_url  , video , near , geo ,source , user_rt_id , user_rt ,retweet_id , reply_to ,retweet_date,sentiment,confidence)' \
          'VALUES("%s","%s","%s", "%s", "%s","%s", "%s", "%s", "%s", "%s","%s", "%s", "%s", "%s", "%s","%s", "%s", "%s", "%s", "%s","%s", "%s", "%s", "%s", "%s","%s", "%s", "%s", "%s", "%s","%s", "%s", "%s")', 
          list(df[i-1:i].values[0]))
    
mydb.commit()




