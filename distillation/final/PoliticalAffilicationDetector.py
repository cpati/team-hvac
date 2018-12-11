
# coding: utf-8

# ## Political Affiliation Factor
# #### This notebook is a subset of PoliticalAffilicationDetector_Complete.ipynb that has lot of other steps like word2vec, tf-idf, LDA I tried

# In[35]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


# In[36]:


columns=['id','label','statement','subject','speaker','speaker_job','state',
        'party_affiliation','barely_true_count','false_Count',
        'half_true_count','mostly_true_count','pants_on_fire_count','venue_speach'];
df_lair=pd.read_csv('../train.tsv',sep='\t',header=None,names=columns,index_col=False);


# In[37]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words=set(stopwords.words('english'))


# ## Text Proprocessing

# #### Function for text preprocessing
# - lowercase the text
# - word tokenization
# - remove stop words and non alphanumeric charaters
# - stemming

# In[38]:


from nltk.stem import PorterStemmer,WordNetLemmatizer
def text_preprocessing(df_base,column):
    df=df_base.copy()
    # lowercase the text
    df[column]=df[column].str.lower()
    # word tokenization
    df[column]=df[column].map(lambda x: nltk.word_tokenize(x))
    # remove stop words and non alphanumeric charaters
    df[column]=df[column].map(lambda x: [w for w in x if (not w in stop_words) and w.isalpha()])
    # lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    df[column]=df[column].map(lambda x: [ wordnet_lemmatizer.lemmatize(w) for w in x])    
    # stemming
    porter = PorterStemmer()
    df[column]=df[column].map(lambda x: [porter.stem(w) for w in x] )
    return df


# #### Calling text_preprocessing function

# In[39]:


df_train=text_preprocessing(df_lair,'statement')


# ## Doc2Vec Political Affilication

# In[40]:


import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec


# In[41]:


vocab_political_affiliation=['president','george','bush','administration','republican','democrats','barack','obama','hillary','clinton','donald','trump','senate','house']


# In[42]:


tagged_doc_pa = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(vocab_political_affiliation)]


# In[43]:


from gensim.models.doc2vec import Doc2Vec
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_doc_pa)

for epoch in range(max_epochs):
    #print('iteration {0}'.format(epoch))
    model.train(tagged_doc_pa,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
    
model.save("d2v.model.pa")  


# In[44]:


df_train_statements_d2v=df_train[['statement','label']]
model= Doc2Vec.load("d2v.model.pa")
texts=[]
for x in df_train_statements_d2v['statement']:
    texts.append(model.infer_vector(x))


# ## Logistics Regression for Doc2Vec 

# In[45]:


def replace_label(x):
    if (x == 'true' or x=='mostly-true' or x=='half-true'):
        return 1
    else:
        return 0
replace_label('true')    


# In[46]:


X=pd.DataFrame(texts)
#y=df_train_statements_d2v[['label']]
y=df_train_statements_d2v['label'].map(lambda x:replace_label(x))
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size = .3, random_state = 1)


# In[47]:


from sklearn.linear_model import LogisticRegression
logisticRegr_D2V_PA = LogisticRegression(C=100)
logisticRegr_D2V_PA.fit(X_train, y_train)
lr_pred_pa = logisticRegr_D2V_PA.predict(X_test)
import pickle
s = pickle.dumps(logisticRegr_D2V_PA)


# In[48]:


from sklearn import metrics
#print(metrics.classification_report(y_test,lr_pred_pa))


# In[49]:


import pickle
def political_affiliation_checker(news):
    data_pred=[]
    data_pred.append(model.infer_vector(news))
    lrg_pa = pickle.loads(s)
    pred_conf=lrg_pa.predict_proba(data_pred)
    #print(pred_conf)
    return pred_conf[0][1]


# In[50]:


class PoliticalAffilicationDetector:
    def __init__(self,news):
        self.news=news
    def predict(self):
        return political_affiliation_checker(self.news)


# In[51]:


#political_affiliation_checker("Says the Annies List political group supports third-trimester #abortions on demand.")


# In[52]:


#logisticRegr_D2V_PA.classes_

