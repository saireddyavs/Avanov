import sqlite3
import pandas as pd
import spacy
import re
import warnings
warnings.filterwarnings("ignore")
nlp = spacy.load('en_core_web_sm')
nlp.Defaults.stop_words.add("virginamerica")
nlp.Defaults.stop_words.add("united")
nlp.Defaults.stop_words.add("unite")
nlp.Defaults.stop_words.add("delta")
nlp.Defaults.stop_words.add("southwest")
nlp.Defaults.stop_words.add("american")
nlp.Defaults.stop_words.add("us airways")
nlp.Defaults.stop_words.add("indigoairline")
nlp.Defaults.stop_words.add("indigo")
nlp.Defaults.stop_words.add("flight")
emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
def preprocess(tweet):
    rev=re.sub('[^a-zA-Z]',' ',tweet)
   
    rev=rev.lower();
    doc = nlp(rev, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in nlp.Defaults.stop_words and tok not in punctuations]
    tokens = ' '.join(tokens)
    twe=emoji_pattern.sub(r'', tokens)
   
    
    return twe
