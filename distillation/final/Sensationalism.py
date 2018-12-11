
# coding: utf-8

# ## Alternus Vera
# Dataset: Politifact/Liar-Liar dataset(https://www.politifact.com)
# 
# Description of the train TSV file format:
# 
# - Column 1: the ID of the statement ([ID].json)
# - Column 2: the label.
# - Column 3: the statement.
# - Column 4: the subject(s).
# - Column 5: the speaker.
# - Column 6: the speaker's job title.
# - Column 7: the state info.
# - Column 8: the party affiliation.
# - Column 9-13: the total credit history count, including the current statement.
#   - 9: barely true counts.
#   - 10: false counts.
#   - 11: half true counts.
#   - 12: mostly true counts.
#   - 13: pants on fire counts.
# - Column 14: the context (venue / location of the speech or statement).

# ------------------------
# - Data Preparation
# - Data exploration
# - Stemming and tokenization
# - Tf-Idf
# - Sentiment Analysis
# - LDA
# - LDA Score calculation and topic inferance 
# **Added tf-idf on bag of words, sentiment analysis, score calculation LDA (on bag of words and tf-idf) and comparision
# 
# -----------------------------
# link to team repo:https://github.com/cpati/team-hvac

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords  
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


df =pd.read_csv('../../liar-liar_dataset/train.tsv', sep='\t')


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[5]:


df.columns


# ##### Data exploration

# In[6]:


df['state'].value_counts()


# In[7]:


df['party'].unique()


# In[8]:


df.loc[df['topic']=='health-care']


# In[9]:


df.isnull().sum()


# ### Data Cleaning
#     Tokenizing: converting a document to its atomic elements.
#     Stopping: removing meaningless words.
#     Stemming: merging words that are equivalent in meaning.

# In[10]:


df1 =df.drop(columns=['file','value' ,'barely true counts', 'false counts',
       'half true counts', 'mostly true counts', 'pants on fire counts',])


# In[11]:


print (df1)


# In[12]:


raw =df1[['statement']]
raw[:5]
final=raw.values.T.tolist()
print (len(final[0]))


# In[13]:


nltk.download('stopwords')


# In[14]:


## porter stemmer
def textProcessing(doc):
    process = re.sub('[^a-zA-Z]', ' ',doc) 
    process = process.lower()
    process = process.split()
    ps = PorterStemmer()
    process = [ps.stem(word) for word in process if not word in set(stopwords.words('english'))]
    process = ' '.join(process)
    return process


# In[15]:


print('original document: ')
words = []
for word in final:
    words.append(word)
print (words)



# In[16]:


nltk.download('stopwords')


# In[17]:


result =[]
for i in final[0]:
    result.append(textProcessing(i))
print (result)


# ### N grams using count vectorization

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer 


# In[19]:


for i in df['statement']:
    vectorizer = CountVectorizer(ngram_range=(1,6))
    analyzer = vectorizer.build_analyzer()
    print (analyzer(i))


# ### tokenization the words

# In[20]:


from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')


# In[21]:


final=[]
for i in result:
    tokens = tokenizer.tokenize(i)
    final.append(tokens)
print (final)



# ### adding all the sentence token's in one list

# In[22]:


import itertools
flat=itertools.chain.from_iterable(final)
text = list(flat)


# In[23]:


from gensim import corpora, models


# In[24]:


dictionary = corpora.Dictionary(final)
print (dictionary)


# In[25]:


import numpy as np
final=np.asarray(final)
raw1 = np.concatenate(final).ravel().tolist()
raw1


# ### Bag of words
# to measure the frequency 

# In[26]:


dictionary.filter_extremes(no_below=20, no_above=0.5, keep_n=100000)
bow_corpus = [dictionary.doc2bow(result) for result in final]
print (bow_corpus[50])



# ### TF-IDF on Bag of words
# to measure the relevance of the words

# In[27]:


from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]


# In[28]:


from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    break


# ### LDA Model on bag of words

# In[29]:


import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
import numpy as np
np.random.seed(2018)


# In[30]:


lda_model = gensim.models.ldamodel.LdaModel(bow_corpus, num_topics=10, id2word = dictionary, passes=2)


# In[31]:


print (lda_model)


# In[32]:


for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# 10 topics:
# - tax rate
# - job
# - voting for bill 
# - immigration
# - President Obama
# - school fund
# - tax on health care

# #### Score calculation

# In[33]:


for index, score in sorted(lda_model[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model.print_topic(index, 10)))


# ### LDA Model on tf -idf

# In[34]:


lda_model_tfidf  = gensim.models.ldamodel.LdaModel(corpus_tfidf, num_topics=10, id2word = dictionary, passes=2)


# In[35]:


print (lda_model)


# In[36]:


for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# 10 topics:
# - tax
# - sex
# - wage
# - college 
# - train
# - job
# - health care
# - Donald Trump
# - oil tax

# #### Score calculation

# In[37]:


for index, score in sorted(lda_model_tfidf[bow_corpus[4310]], key=lambda tup: -1*tup[1]):
    print("\nScore: {}\t \nTopic: {}".format(score, lda_model_tfidf.print_topic(index, 10)))



# ### Sentiment Analysis

# polarity: negative, neutral, positive, compound
# polarity annotations in output: 
# - negative - neg
# - neutral - neu
# - positive - pos
# - compound - compound

# In[38]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


# In[39]:


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    #print("{}{}".format(sentence, str(score)))
    return score


# In[40]:


polarity=[]
for i in df['statement']:
    result = (sentiment_analyzer_scores(i))
    polarity.append(result)
    print (i)
    print (result)
    print ("")
    


# In[41]:


range(len(polarity))


# In[42]:


negative=[]
positive=[]
neutral=[]
compound=[]
size = len(polarity)
k=0
while k < len(polarity):
    negative.append(polarity[k].get('neg'))
    neutral.append(polarity[k].get('neu'))
    positive.append(polarity[k].get('pos'))
    compound.append(polarity[k].get('compound'))
    k=k+1


# In[43]:


df['negative']= negative
df['positive']=positive
df['neutral']= neutral
df['compound']=compound


# In[44]:


df.columns


# In[45]:


df.head(10)


# #### Final Sentiment

# In[46]:


df1 = pd.concat([df['positive'] ,df['negative'],df['compound'],df['neutral'] ], axis=1, join='outer')

print (df1.head(10))
#df1 =pd.DataFrame(np.array(list1).reshape(4,10240))
#print (df1.head())


# In[47]:


print (df1.hist())


# It can be observed that maximum amount of statements are neutral as its compound value is 0, negative is 0, positive is 0 and neutral is 1.

# In[48]:


neg = df['negative']
neu =df['neutral']
comp = df['compound']
pos =df['positive'] 
print ("Attribute","Positive","Negative","Compound","Neutral" )
print ("Max: \t", "{0:.2f} \t".format(pos.max()), "{0:.2f} \t".format(neg.max()), "{0:.2f}\t".format(comp.max()), "{0:.2f}\t".format(neu.max()))
print ("Min: \t", "{0:.2f} \t".format(pos.min()), "{0:.2f} \t".format(neg.min()), "{0:.2f}\t".format(comp.min()), "{0:.2f}\t".format(neu.min()))
print ("Avg: \t", "{0:.2f} \t".format(pos.mean()),"{0:.2f} \t".format(neg.mean()),"{0:.2f}\t".format(comp.mean()),"{0:.2f}\t".format(neu.mean()))
print ("Std.Devi", "{0:.2f} \t".format(pos.std()),  "{0:.2f} \t".format(neg.std()), "{0:.2f}\t".format(comp.std()), "{0:.2f}\t".format(neu.std()))


# ### Inference
# -- Based on the information gained by skimming and scanning through different Fake news article and our dataset is that:
# 1. It can be said the statments which has the following sentiments are not fake news:
# - compound = 0, 
# - negative= 0, 
# - positive = 0,
# - neutral =  1
# 
# 2. Fake news contains highly positve emotions or negative emotions.
# Compound Score: The Compound score is a metric that calculates the sum of all the lexicon ratings which have been normalized between -1(most extreme negative) and +1 
# - In our case we also have compound value. Negative emotions have compound value closure to -1 and positive have closure to 1.
# 
# #### Hence, we can conclude the below formula:
# - if positive value is >0.75 or negative value is >0.75 or compound value is between (-0.75,0.75) can be suspected as fake news

# In[49]:


### test function (non used)
value =0
for i in range(len(df)):
    if (df['negative'].iloc[i] > 0.75):
        if (df['compound'].iloc[i] < 0.75):
            value= 60*(df['negative'].iloc[i]+df['compound'].iloc[i])
            df['value'].iloc[i]=value
        else:
            value = 60* df['negative'].iloc[i]
            df['value'].iloc[i]=value
    elif (df['positive'].iloc[i]>0.75):
        if(df['compound'].iloc[i]>0.75):
            value= 40*(df['positive'].iloc[i]+df['compound'].iloc[i])
            df['value'].iloc[i]=value
        else:
            value = 40*df['positive'].iloc[i]
            df['value'].iloc[i]=value


# In[83]:


### main sensationalism function
def Sensationalism(sentence):
    score = analyser.polarity_scores(sentence)
    final={}
    negative =score.get('neg')
    positive =score.get('pos')
    compund = score.get('compound')
    neutral =score.get('neu')
    final['negative'] = negative
    final['positive'] = positive
    final['compund'] = compund
    final ['neutral'] = neutral
    if (neutral == 1):
        value = 0
        
    elif (negative>0.7):
        value = negative
        
    elif (positive>0.7):
        value = positive
        
    elif ( neutral>0.65):
        value = (1-neutral)
        
    elif ( negative<0.7 and  positive<0.7 and (positive - negative>0)):
        value = (positive - negative)
        
    elif (  negative<0.7 and  positive<0.7 and  negative>0.4 and (positive - negative<0.5)  ):
        if(abs(compund)>negative):
            value = (1- abs(compund))
            
        else:
            value = (1- negative)
            
    else:
        value = abs(compund)
    final['sensationl_value'] = float('{:,.3f}'.format(value))
    #return final
    return value


# In[84]:


sensational=[]
for i in df['statement']:
    result = (Sensationalism(i))
    sensational.append(result)
    print (i)
    print (result)
    print ("")


# In[85]:


plt.hist(sensational)


# In[86]:


test = Sensationalism('It was under Barack Obama and Hillary Clinton that changed the rules of engagement that probably cost (Capt. Humayun Khans) life.')
print (test)


# ##  Applying sensationalism function on testing dataset

# In[55]:


dfTest =pd.read_csv('../../liar-liar_dataset/test.tsv', sep='\t')


# In[56]:


dfTest.head(10)


# In[57]:


dfTest['statement'].head(10)


# In[87]:


polarityTest=[]
for i in dfTest['statement']:
    result = (Sensationalism(i))
    polarityTest.append(result)
    print (i)
    print (result)
    print ("")


# In[74]:


plt.hist(polarityTest)


# In[88]:


class Sensationalism:
    def __init__(self,news):
        self.news=news
    def predict(self):
        return Sensationalism(self.news)

