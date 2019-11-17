# -*- coding: utf-8 -*-
"""oneagain.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1L9s6lFM2CJiX49LdCSnBZxMqGPUcGZYw
"""
def venkat():
    import pandas as pd
    papers = pd.read_excel('nag.xlsx')

    X = papers.iloc[:,3]
    '''
    # Load the regular expression library
    import re
    # Remove punctuation
    papers['comments'] = papers['comments'].map(lambda x: re.sub('[!"#$%&\'()*+,-/:;<=>?@\\^_`{|}~]', '', x))
    # Convert the titles to lowercase
    papers['comments'] = papers['comments'].map(lambda x: x.lower())
    # Print out the first rows of papers
    papers['comments'].head()
    '''
    import spacy
    import re
    nlp = spacy.load('en')
    nlp.Defaults.stop_words.add("etihadairways")
    nlp.Defaults.stop_words.add("etihadairway")
    nlp.Defaults.stop_words.add("etihad")
    nlp.Defaults.stop_words.add("airway")
    nlp.Defaults.stop_words.add("etihadairlines")

    nlp.Defaults.stop_words.add("send")
    nlp.Defaults.stop_words.add("etihadhelp")
    import re
    j=[]
    for i in X:
      i = str(i)
      emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
      j.append(emoji_pattern.sub(r'', i)) # no emoji
    #print(j)
    j = list(j)

    j = pd.DataFrame(j)
    #print(j)
    j.iloc[:,0] = j.iloc[:,0].replace(r'https\S+', '', regex=True).replace(r'http\S+', '', regex=True)
    j = j[0]
    #print(j.head())
    #print(j.shape)

    j = j.astype(str)
    punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~'
    texts = []
    for i in range(papers.shape[0]):
        rev=re.sub('[^a-zA-Z]',' ',j[i])
        rev=rev.lower();
        doc = nlp(rev, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in nlp.Defaults.stop_words and tok not in punctuations]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    f = pd.DataFrame(texts)
    #print(f.head())
    #print(f.values)

    '''
    # Import the wordcloud library
    from wordcloud import WordCloud
    # Join the different processed titles together.
    long_string = ','.join(list(f[0].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud
    wordcloud.to_image()
    '''



    # Load the library with the CountVectorizer method
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np
    #import matplotlib.pyplot as plt
    #import seaborn as sns
    #sns.set_style('whitegrid')
    #%matplotlib inline
    # Helper function
    z1=[]
    def plot_10_most_common_words(count_data, count_vectorizer):
        import matplotlib.pyplot as plt
        words = count_vectorizer.get_feature_names()

        total_counts = np.zeros(len(words))
        for t in count_data:
            total_counts+=t.toarray()[0]

        count_dict = (zip(words, total_counts))
        count_dict = sorted(count_dict, key=lambda x:x[1], reverse=True)[0:20]
        words = [w[0] for w in count_dict]
        counts = [w[1] for w in count_dict]
        x_pos = np.arange(len(words))

        #plt.figure(2, figsize=(15, 15/1.6180))
        #plt.subplot(title='10 most common words')
        #sns.set_context("notebook", font_scale=1.25, rc={"lines.linewidth": 2.5})
        #sns.barplot(x_pos, counts, palette='husl')
        #plt.xticks(x_pos, words, rotation=90)
        #plt.xlabel('words')
        z1.append(words)
        #plt.ylabel('counts')
        #plt.show()
    # Initialise the count vectorizer with the English stop words
    count_vectorizer = CountVectorizer(stop_words='english')
    # Fit and transform the processed titles
    count_data = count_vectorizer.fit_transform(f[0])
    # Visualise the 10 most common words
    plot_10_most_common_words(count_data, count_vectorizer)
    #print(l)

    import warnings
    warnings.simplefilter("ignore", DeprecationWarning)
    # Load the LDA model from sk-learn
    from sklearn.decomposition import LatentDirichletAllocation as LDA
    x=[]
    # Helper function
    words = count_vectorizer.get_feature_names()
    for topic_idx, topic in enumerate(lda.components_):
        #print("\nTopic #%d:" % topic_idx)
        x.append([words[i]
                        for i in topic.argsort()[:-number_words - 1:-1]])

    # Tweak the two parameters below
    number_topics = 5
    number_words = 8
    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)
    # Print the topics found by the LDA model

    import nltk
    nltk.download('averaged_perceptron_tagger')
    selective_pos = ['NN','NNS','NNP','NNPS']
    z = []
    for i in z1:
      for word,tag in nltk.pos_tag(i):
          if tag in selective_pos:
              z.append((word))
    return [x,z]


