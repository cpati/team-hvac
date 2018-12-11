
# coding: utf-8

# ## Stance Detection Factor
# #### This notebook is a subset of Spam_detection_and_Stance_detection.ipynb

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# #### Dataset is used from fakenewschallenge

# In[2]:


dataset_stances2 = pd.read_csv('../train_stances.csv', sep=',')
dataset_body2 = pd.read_csv('../train_bodies.csv', sep=',')
UCI_Aggregator_load = pd.read_csv('../uci-news-aggregator.csv',sep=',')
UCI_Aggregator = UCI_Aggregator_load.dropna(how='any')


# #### Fakeness impact based on the claim stance

# * agree = 0 (If most of the claims agree with the news, the news is possibly not fake)
# * disagree = 1 (If most of the claims disagree with the news, the news is possibly fake)
# * discuss = 0.5 (If there is a lot of discussion happening around, the news may or may not be fake)
# * unrealted = 1 (If the stances are mostly unrealted, the news will most likely be fake)

# #### Create a doc2vec model for the training dataset

# In[3]:


from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(dataset_body2['articleBody'])]
#print (tagged_data)


# In[4]:


max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =0)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    #print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha
model.save("doc2vec.model")
#print("Model Saved")


# #### Once the model is trained load it to use it on the testing dataset

# In[5]:


from gensim.models.doc2vec import Doc2Vec
model= Doc2Vec.load("doc2vec.model")


# In[6]:


dataset_body2['index'] = dataset_body2.index


# #### This python fucntion will scrape the fakenewschallenge dataset to get claims for all the news articles present in new aggregator dataset

# #### Once the relevant claims are found a new dataset will be returned which will have just the required details like news article and respective claims

# In[7]:


def get_claims():
    from gensim.models.doc2vec import Doc2Vec
    model= Doc2Vec.load("doc2vec.model")
    dataset_body2['index'] = dataset_body2.index
    count=1
    uci_news=0
    fnc_news_body_index=0
    for i in range(14204, 400000):
        claim1=UCI_Aggregator['TITLE'][i]
        claim1.lower()
        new_test_data = word_tokenize(claim1.lower())
        v2 = model.infer_vector(new_test_data)
        similar_documents = model.docvecs.most_similar([v2], topn = 1)
        myarray = np.asarray(similar_documents)
        new_a=myarray.squeeze()
        similarity_score=new_a[1]
        article_id = new_a[0]
        if float(similarity_score) > 0.95:
            #print ("{}".format(count))
            #print ("At index {}, Claim - {}".format(i, claim1))
            uci_news = i
            #print("** Similar news articles **")
            #print(dataset_body2.loc[dataset_body2['index'] == int(new_a[0]), 'articleBody'])
            fnc_news_body_index = int(new_a[0])
            count+=1
    UCI_Aggregator['TITLE'][uci_news]
    dataset_body2['articleBody'][fnc_news_body_index]
    Body_ID = dataset_body2.loc[dataset_body2['index'] == fnc_news_body_index, 'Body ID']
    new_df = dataset_stances2[dataset_stances2['Body ID'] == int(Body_ID)]
    new_df2 = new_df.join(new_df['Stance'].str.get_dummies())
    return new_df2


# #### Now we will analyse the stances of all the claims from the new dataset

# In[8]:


def stance_check():
    df = get_claims()
    arr=df['Stance'].value_counts().index
    for i in arr:
        if i == 'discuss':
            Stance_check=0.5
        elif i == 'agree':
            Stance_check=0
        elif i == 'disagree':
            Stance_check=1
        else:
            Stance_check=1
    return Stance_check
#stance_check()


# In[9]:


class StanceDetection:
    def __init__(self,news):
        self.news=news
    def predict(self):
        return stance_check()

