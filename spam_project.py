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

print('package imported...')

df=pd.read_csv('dataset/sms.txt',delimiter='\t',names=['isSpam','msg'])
punc=string.punctuation
stop_words=stopwords.words('english')
stop_words.remove('not')
ps=PorterStemmer()
cv=CountVectorizer(binary=True)
pca=PCA(.99)
log=LogisticRegression()

print('object created...')

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

df['msg']=df.msg.apply(clean_text)

print('data cleaned...')

X=cv.fit_transform(df.msg).toarray()
new_X=pca.fit_transform(X)
y=df.iloc[:,0].values
print('going for training...')

log.fit(new_X,y)
print('model trained....')
msg2=input('enter msg:')
msg2=clean_text(msg2)
test_x=cv.transform([msg2]).toarray()
test_x=pca.transform(test_x)
pred=log.predict(test_x)
print(pred[0])
