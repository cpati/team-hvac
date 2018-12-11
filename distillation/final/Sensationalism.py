
# coding: utf-8

# In[4]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[5]:


def Sensationalism(sentence):
    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(sentence)
    negative =score.get('neg')
    positive =score.get('pos')
    compund = score.get('compound')
    neutral =score.get('neu')
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
    return value


# In[6]:


class Sensationalism:
    def __init__(self,news):
        self.news=news
    def predict(self):
        return Sensationalism(self.news)

