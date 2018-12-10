
# coding: utf-8

# ## Spam Detection Factor
# #### This notebook is a subset of Spam_detection_and_Stance_detection.ipynb

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv('../himangini_train.tsv', sep='\t')
data_text=data[['statement']]
data_text['index'] = data_text.index
documents = data_text


# In[3]:


spam_dataset = pd.read_csv("../spam.csv", encoding = "latin-1")
spam_dataset = spam_dataset[['v1', 'v2']]
spam_dataset = spam_dataset.rename(columns = {'v1': 'label', 'v2': 'text'})


# In[4]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
lemmatizer = WordNetLemmatizer()
stopwords = set(stopwords.words('english'))
def review_messages(msg):
    # converting messages to lowercase
    msg = msg.lower()
    return msg

def alternative_review_messages(msg):
    # converting messages to lowercase
    msg = msg.lower()

    # removing stopwords 
    msg = [word for word in msg.split() if word not in stopwords]

    # uses a lemmatizer
    msg = " ".join([lemmatizer.lemmatize(word) for word in msg])
    return msg


# In[5]:


spam_dataset['text'] = spam_dataset['text'].apply(review_messages)
X_train, X_test, y_train, y_test = train_test_split(spam_dataset['text'], spam_dataset['label'], test_size = 0.1, random_state = 1)


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
from sklearn import svm
svm = svm.SVC(C=1000)
svm.fit(X_train, y_train)


# In[7]:


def spam_checker(msg):
    msg = vectorizer.transform([msg])
    prediction = svm.predict(msg)
    return prediction[0]
#spam_checker("Says the Annies List political group supports third-trimester abortions on demand.")


# In[8]:


class SpamDetection:
    def __init__(self,news):
        self.news=news
    def predict(self):
        return spam_checker(self.news)

