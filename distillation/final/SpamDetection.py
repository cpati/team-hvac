
# coding: utf-8

# ## Spam Detection Factor
# #### This notebook is a subset of Spam_detection_and_Stance_detection.ipynb

# In[10]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# #### Load the liar liar dataset

# In[11]:


data = pd.read_csv('../himangini_train.tsv', sep='\t')
data_text=data[['statement']]
data_text['index'] = data_text.index
documents = data_text


# #### Using SMS Spam Detection Kaggle dataset

# In[12]:


spam_dataset = pd.read_csv("../spam.csv", encoding = "latin-1")
spam_dataset = spam_dataset[['v1', 'v2']]
spam_dataset = spam_dataset.rename(columns = {'v1': 'label', 'v2': 'text'})


# In[13]:


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


# #### Splitting the model into test and train 

# In[14]:


spam_dataset['text'] = spam_dataset['text'].apply(review_messages)
X_train, X_test, y_train, y_test = train_test_split(spam_dataset['text'], spam_dataset['label'], test_size = 0.1, random_state = 1)


# #### Using Support Vector Machine to build a spam/ham classifier model

# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
from sklearn import svm
svm = svm.SVC(C=1000)
svm.fit(X_train, y_train)


# #### Using the trained model on liar liar dataset, the below python function will check any news and return ham or spam as per the classification

# In[16]:


def spam_checker(msg):
    msg = vectorizer.transform([msg])
    prediction = svm.predict(msg)
    return prediction[0]


# #### Now, lets have a python function which can give us the numerical value for alternus vera computation

# In[17]:


ham1 = "ham"
def spam_detection_checker(msg):
    spam_value = spam_checker(msg)
    if spam_value == ham1:
        return 0
    else:
        return 1


# In[18]:


class SpamDetection:
    def __init__(self,news):
        self.news=news
    def predict(self):
        return spam_detection_checker(self.news)

