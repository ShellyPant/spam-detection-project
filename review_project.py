import nltk
import string
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('dataset/Restaurant_Reviews.txt',delimiter='\t')
punc=string.punctuation
stop_words=stopwords.words('english')
stop_words.remove('not')
ps=PorterStemmer()
cv=CountVectorizer(binary=True)
pca=PCA(.99)
log=LogisticRegression()

def clean_text(msg):
    msg=msg.lower()
    msg=re.sub(f'[{punc}]','',msg)
    words=word_tokenize(msg)
    new_words=[]
    for w in words:
        if(w not in stop_words):
            new_words.append(w)
    
    after_stem_words=[]
    for w in new_words:
        after_stem_words.append(ps.stem(w))
    clean_msg=' '.join(after_stem_words)
    return clean_msg

df['Review']=df.Review.apply(clean_text)
X=cv.fit_transform(df.Review).toarray()
new_X=pca.fit_transform(X)
y=df.iloc[:,-1].values
log.fit(new_X,y)
print('model trained....')
msg=input('enter review:')
msg=clean_text(msg)
test_x=cv.transform([msg]).toarray()
test_x=pca.transform(test_x)
pred=log.predict(test_x)
if(pred[0]==0):
    print('not like')
if(pred[0]==1):
    print('like')