
# coding: utf-8

# In[ ]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
stop_words=set(stopwords.words('english'))


# In[ ]:


from nltk.corpus import wordnet
import enchant
enchant = enchant.Dict("en_US")
def spell_check_article(article):
    fake=0
    article=article.replace('-',' ');
    article_tokenized=nltk.word_tokenize(article)
    total_word_count=len(article_tokenized)
    words_with_error_count=0
    for w in article_tokenized:
        if (not wordnet.synsets(w)) and (not enchant.check(w)):
            words_with_error_count+=1
            if words_with_error_count > 3:
                fake=1
    return fake


# In[ ]:


class SpellChecker:
    def __init__(self,news):
        self.news=news
    def predict(self):
        return spell_check_article(self.news)

