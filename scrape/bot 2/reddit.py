# -*- coding: utf-8 -*-
"""reddit.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1P5_vXuleTGCeaSMMWvwD1ggBm4pVQfMX

**install dependency**
"""

!pip install praw

"""**import praw**"""

import praw

"""**let's  be open source follow below link to get your client id,secret key,user_agent**

1.   visit https://gilberttanner.com/blog/scraping-redditdata 

2.   get your client id,client secret,user_agent and back to below code.
"""



reddit = praw.Reddit(client_id='JIWjSCiTyNIS9w', client_secret='9YUNrUpOgVeJUbRyWhRpEDaofUU', user_agent='babyop')

"""**enter keyword to search post and extract comments of user with username**"""

print("enter list of keywords:")
b = input()
#b = list(b.split(" "))
#print(b)
import spacy
import nltk
nltk.download('averaged_perceptron_tagger')
  
nlp = spacy.load('en')


import re
l=[]
k=[]
d=[]
cnt=0

punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
texts = []
#for i in b:
rev=re.sub('[^a-zA-Z]',' ',str(b))
rev=rev.lower();
doc = nlp(rev, disable=['parser', 'ner'])
tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
tokens = [tok for tok in tokens if tok not in nlp.Defaults.stop_words and tok not in punctuations]
#tokens = ' '.join(tokens)
#texts.append(tokens)
tagged = nltk.pos_tag(tokens)
print(tagged)
a=[]
for i in tagged:
    log = (i[1][0] == 'N')
    if log == True:
      a.append(i[0])
print("this is a",a)
sk = set(tokens)
sk = list(sk)
print("this is sk",sk)
print(tokens)
for i in sk:
  try:
    from praw.models import MoreComments
    hot_posts = reddit.subreddit(i).hot(limit=500)
    for post in hot_posts:
        submission = reddit.submission(id=post)
        vk = post.title
        punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
        texts = []
        #for i in b:
        rev=re.sub('[^a-zA-Z]',' ',str(vk))
        rev=rev.lower();
        doc = nlp(rev, disable=['parser', 'ner'])
        sent = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        sent = [tok for tok in sent if tok not in nlp.Defaults.stop_words and tok not in punctuations]
        #sent = set(sent)
        #print("this is sent",sent)
        #sk.remove(i)
        #print(set(sent), set(sk))
        if len(sk)==1:
          x=0
        else:
          x=1
        if len(set.intersection(set(sent), set(sk)))>x:
          print(set.intersection(set(sent), set(sk)))
          for top_level_comment in submission.comments:
            for top_level_comment in submission.comments:
              if isinstance(top_level_comment, MoreComments):
                continue
              l.append(top_level_comment.body)
              #k.append(top_level_comment.id)
              d.append(top_level_comment.author)
              cnt+=1
  except:
    print(i,"not valid")
  print(i,cnt)
  cnt=0

"""**make Dataframe and print Data**"""

import pandas as pd
comment = pd.DataFrame(l)
user = pd.DataFrame(d)
#print(j.head())
#print(j.shape)
#print(j1.shape)
#print(j1.head())
x = pd.concat([user, comment], axis=1)
x.columns = ['username','comments']
x.to_csv("reddit2.csv")
print(x.head())
print(x.shape)

import sqlite3
dbname = 'redditdata2'
conn = sqlite3.connect(dbname + '.sqlite')
cur = conn.cursor()
import pandas as pd
#if we have a csv file
df = pd.read_csv('reddit2.csv',sep=',')
df.to_sql(name='redtable', con=conn)
cur.execute('SELECT * FROM redtable')

for i in cur:
  print(i)


