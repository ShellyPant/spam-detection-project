from tkinter import *
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
from sklearn.naive_bayes import BernoulliNB
print('package imported...')

df=pd.read_csv('dataset/sms2.txt',delimiter='\t',names=['isSpam','msg'])
punc=string.punctuation
stop_words=stopwords.words('english')
stop_words.remove('not')
ps=PorterStemmer()
cv=CountVectorizer(binary=True)
pca=PCA(.99)
#log=LogisticRegression()
log=BernoulliNB()

print('objects created...')

def mypredict():
	msg=e.get()
	msg2=clean_text(msg)
	test_x=cv.transform([msg2]).toarray()
	test_x=pca.transform(test_x)
	pred=log.predict(test_x)
	l3.configure(text=pred[0])

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
	

root=Tk()
root.state('zoomed')
root.configure(background='yellow')
l1=Label(root,text='Spam Detection',bg='yellow',fg='blue',font=('',40,'bold'))
l1.place(x=190,y=20)

l2=Label(root,text='Enter msg:',bg='yellow',fg='blue',font=('',20,'bold'))
l2.place(x=150,y=220)

e=Entry(root,font=('',20,''))
e.place(x=310,y=220)

b=Button(root,text='Predict',font=('',20,''),command=mypredict)
b.place(x=380,y=280)

l3=Label(root,text='',bg='yellow',font=('',20,'bold'))
l3.place(x=380,y=380)

root.mainloop()