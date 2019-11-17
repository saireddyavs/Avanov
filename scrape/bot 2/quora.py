
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import csv
import pandas as pd
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))

def scrapper_quora(sentence):
    
    import nltk
    nltk.download('averaged_perceptron_tagger')

    

    import re
    l=[]
    k=[]
    d=[]
    cnt=0
    manylinks=[]
    usernames=[]

    punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
    texts = []
    #for i in b:
    rev=re.sub('[^a-zA-Z]',' ',str(sentence))
    rev=rev.lower();
    
    
    tokens = [tok for tok in rev.split() if tok not in stop and tok not in punctuations]
    #tokens = ' '.join(tokens)
    #texts.append(tokens)
    '''
    tagged = nltk.pos_tag(tokens)
    print(tagged)
    a=[]
    for i in tagged:
        log = (i[1][0] == 'N')
        if log == True:
          a.append(i[0])
    print("this is a",a)
    '''
    sk = set(tokens)
    print("this is sk",sk)
    print(tokens)
    
    for i in sk:
        manylinks.append("https://www.quora.com/topic/"+str(i))
    
    
#     link1 = "https://www.quora.com/topic/"+keyword
#     #link2 = input("Enter second link")
#     manylinks = list()
#     manylinks.append(link1)
    #manylinks.append(link2)
    
    ques=[]
    ques_no=[]
   
    
    qusers=[]
    user_details=[]

    
    ful=[]
    ans_no=[]
    ans=[]
    ful1=[]
    count=1
    
    for olink in manylinks:
        qlinks = list()
        print(olink)
        options = webdriver.ChromeOptions()
        options.binary_location = "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"
        browser = webdriver.Chrome(executable_path=r'C:\Users\saireddyavs\Desktop\twitter bot\twitter_scrapper\chromedriver', chrome_options=options)
        try:
            browser.get(olink)
        except:
            continue
        time.sleep(1)
        elem = browser.find_element_by_tag_name("body")


        no_of_pagedowns = 50
        while no_of_pagedowns:
            elem.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.2)
            no_of_pagedowns-=1
        post_elems =browser.find_elements_by_xpath("//a[@class='question_link']")
        for post in post_elems:
            qlink = post.get_attribute("href")
            print(qlink)
            qlinks.append(qlink)
        
        print(len(qlinks))

        for qlink in qlinks:

            append_status=0

            row = list()
            
            us=[]
            
            

            browser.get(qlink)
            time.sleep(1)


            elem = browser.find_element_by_tag_name("body")


            no_of_pagedowns = 1
            while no_of_pagedowns:
                elem.send_keys(Keys.PAGE_DOWN)
                time.sleep(0.2)
                no_of_pagedowns-=1


    #         #Question Names
    #         qname =browser.find_elements_by_xpath("//div[@class='question_text_edit']")
    #         for q in qname:
    #             print(q.text)
    #             row.append(q.text)
            qname =browser.find_elements_by_xpath("//div[@class='question_text_edit']")
            k1=0
            k2=0
            for q in qname:
                
                
                for i in sk:
                    if i in q.text.lower():
                        k1+=1
                        
                if k1=len(sk):
                    qq=q.text
                    ful.append([qq,qlink])
                    print(q.text)
                    
                

            all_ans=browser.find_elements_by_xpath("//div[@class='ui_qtext_expanded']")
            i=1
            answer_field = list()
            for post in all_ans:
                if True:
                    
                    
                    
                    for i in sk:
                        if i in post.text.lower():
                            k2+=1
                    if k2=len(sk):
#                         row.append(post.text)
#                         ans.append(post.text)
#                         ans_no.append("answer to question_"+str(count))
                        print("Answer : ")
                        print(post.text)

                        ful.append([post.text,qlink])
                
                        
        
                    
                    
                else:
                    break

            
            
            if k1<len(sk) and k2<len(sk):
                continue
            
                
            try:
                
                name=browser.find_elements_by_xpath("//a[@class='user']")
                
                for i in name:
                    print(i.text)
                    usernames.append([i.text,qlink])
                    us.append(i.text)
                    
                
               
            except:
                name=""
                print("name not came")
            
            
            
            
            
            
            


           # print(row)
            

            

            print("*"*50)
            count+=1

        browser.quit() 

    df=pd.DataFrame(ful,columns=["text","link"])
    df.to_csv("questions.csv")
#         df=pd.DataFrame(ful1,columns=['answer',"answers_number"])
#         df.to_csv("answers.csv")

        
        
        
        
    user_details=[]
    for i in usernames:
            user_details.append([info(i[0]),i[1]])
        
    users=[]
    for i in user_details:
            print(i)
            l=list(i[0])
            l.append(i[1])
            users.append(l)

    df=pd.DataFrame(users,columns=["name","description","education","lives_in","num_answers","num_questions","num_followers","num_following","link"])


    df.fillna("")

    df.to_csv("user_details.csv")

        
                  



def info(name):
        driver = webdriver.Chrome(executable_path='chromedriver')
        s="";
        for i in name.split():
            s+="-"+i
        import re
        try:
            driver.get("https://www.quora.com/profile/"+s[1:])
        except:
            return
        try:
            description = driver.find_element_by_class_name('UserCredential').text
        except:
            description=""
        try:
            credentials_and_highlights = driver.find_element_by_class_name('AboutSection')
        except:
            credentials_and_highlights=""
            
            
        try:
                education = credentials_and_highlights.find_element_by_class_name('SchoolCredentialListItem')
                education = re.sub(r'Studied at', '', education.find_element_by_class_name('UserCredential').text).strip()
        except:
                education = ''
        try:
                lives_in = credentials_and_highlights.find_element_by_class_name('LocationCredentialListItem')
                lives_in = re.sub(r'Lives in', '', lives_in.find_element_by_class_name('UserCredential').text).strip()
        except:
                lives_in = ''
       
        print(name)
        print(description)
        print(education)
        print(lives_in)
        
        try:
            num_answers = int(re.sub('\D', '', driver.find_element_by_class_name('AnswersNavItem').find_element_by_class_name('list_count').text))
        except:
            num_answers=0
        try:
            num_questions = int(re.sub('\D', '', driver.find_element_by_class_name('QuestionsNavItem').find_element_by_class_name('list_count').text))
        except:
            num_questions=0
        
        try:
            num_followers = int(re.sub('\D', '', driver.find_element_by_class_name('FollowersNavItem').find_element_by_class_name('list_count').text))
        except:
            num_followers=0
        try:
            num_following = int(re.sub('\D', '', driver.find_element_by_class_name('FollowingNavItem').find_element_by_class_name('list_count').text))
        except:
            num_following=0
        try:
           num_topics = int(re.sub('\D', '', driver.find_element_by_class_name('TopicsNavItem').find_element_by_class_name('list_count').text))
        except:
            num_topics=0

 
            
            
        print(name,description)
        print(num_answers,num_questions,num_followers,num_following)
        
        return name,description,education,lives_in,num_answers,num_questions,num_followers,num_following



        driver.quit()



scrapper_quora(input("Enter sentence::::"))

question=pd.read_csv("questions.csv")
question = question.loc[:, ~question.columns.str.contains('^Unnamed')]

question.isnull().any()
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



question['cleaned']=question.text.apply(lambda x:preprocess(x))



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


q=predict_lstm(list(question.cleaned))
question['sentiment']=q[0]


import numpy as np

question['confidence']=q[1].tolist()


question.drop('cleaned', axis=1, inplace=True)

question = question.astype(str)

import mysql.connector as mysql
# question=pd.read_csv("question.csv")
# question = question.loc[:, ~question.columns.str.contains('^Unnamed')]
mydb = mysql.connect(
  host="127.0.0.1",
  user="sai",
  passwd="rgukt123",
    port="3309"
)

print(mydb)


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

#sql = "DROP TABLE IF EXISTS Quora"

#mycursor.execute(sql) 

#mycursor.execute("CREATE TABLE Quora( question text, link text,sentiment varchar(20),confidence text)")



for i in question.index:
    if i==0:
        continue
    
    mycursor.execute('INSERT INTO Quora(question , link, sentiment,confidence)' 'VALUES("%s", "%s", "%s","%s")', list(question[i-1:i].values[0]))



import pandas as pd
df=pd.read_csv("user_details.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]


import mysql.connector as mysql
df=pd.read_csv("user_details.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
mydb = mysql.connect(
  host="127.0.0.1",
  user="sai",
  passwd="rgukt123",
    port="3309"
)

print(mydb)


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

sql = "DROP TABLE IF EXISTS Quora_User_Details"

mycursor.execute(sql) 

mycursor.execute("CREATE TABLE Quora_User_Details( name text, description text, education text, lives_in text, num_answers text,num_questions text , num_followers text, num_following text,link text)")



for i in df.index:
#     print(i)
    if i==0:
        continue
    
    mycursor.execute('INSERT INTO Quora_User_Details(name , description, education , lives_in , num_answers ,num_questions , num_followers , num_following,link)' \
          'VALUES("%s", "%s","%s","%s","%s","%s","%s","%s","%s")', 
          list(df[i-1:i].values[0]))
mydb.commit()






