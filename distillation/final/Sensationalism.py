
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def Sensationalism_score(sentence):
    analyser = SentimentIntensityAnalyzer()
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




#test = Sensationalism('A strong bipartisan majority in the House of Representatives voted to defund Obamacare.')
#print (test)

class Sensationalism:
    def __init__(self,news):
        self.news=news
    def predict(self):
        return Sensationalism_score(self.news)

