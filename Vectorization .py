#!/usr/bin/env python
# coding: utf-8

# In[95]:


import numpy as np
import nltk
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re


import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud
from tensorflow.keras.layers import Dense,Embedding,LSTM,Conv1D,MaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer


# In[96]:


import warnings
warnings.filterwarnings('ignore')


# In[116]:


fake = pd.read_csv(r'Fake.csv').sample(5000,random_state=10)
real = pd.read_csv(r'True.csv').sample(5000,random_state=10)


# In[119]:


fake.shape


# In[120]:


real.shape


# In[121]:


fake.head()


# In[122]:


fake['subject'].value_counts()


# In[123]:


real.head(2)


# In[124]:


real['subject'].value_counts()


# In[125]:


#Word Cloud of Fake News

text = " ".join(fake['text'].tolist())


# In[126]:


plt.figure(figsize=(10,20))
wordcloud = WordCloud(width=1500,height=1000, prefer_horizontal = 0.5, background_color="rgba(255, 255, 255, 0)",mode="RGBA").generate(text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show();


# In[127]:


text = " ".join(real['text'].tolist())

plt.figure(figsize=(10,20))
wordcloud = WordCloud(width=1800,height=1200, prefer_horizontal = 0.5, background_color="rgba(255, 255, 255, 0)",mode="RGBA").generate(text)
plt.imshow(wordcloud)
plt.axis('off')
plt.show();


# In[128]:


real.sample(5)


# In[129]:


unknown_publishers = []
for index,row in enumerate(real.text.values):
    try:
        record = row.split('-',maxsplit = 1)
        record[1]
        assert(len(record[0])<120)  ## if char is less than 120 than it's tweet
    except:
        unknown_publishers.append(index)

len(unknown_publishers)


# In[130]:


record[0]


# In[131]:


real.iloc[unknown_publishers].text


# In[132]:


real = real.drop(8970,axis=0)


# In[135]:


publisher = []
temp_text = []

for index,row in enumerate(real.text.values):
    if index in unknown_publishers:
        temp_text.append(row)
        publisher.append('Unknown')
    else:
        record = row.split('-',maxsplit=1)
        publisher.append(record[0].strip())
        temp_text.append(record[1].strip())


# In[136]:


real['publisher'] = publisher
real['text'] = temp_text


# In[21]:


real.head()


# In[138]:


empty_fake_index = [index for index,text in enumerate(fake.text.tolist()) if str(text).strip()==""]


# In[139]:


fake.iloc[empty_fake_index]


# In[140]:


fake['text'] = fake['title'] + " " + fake['text']
real['text'] = real['title'] + " " + real['text']


# In[141]:


fake.iloc[empty_fake_index]


# In[142]:


real['text'] = real['text'].apply(lambda x: str(x).lower())
fake['text'] = fake['text'].apply(lambda x: str(x).lower())


# In[143]:


real['class'] = 1
fake['class'] = 0


# In[144]:


real = real[['text','class']]
fake = fake[['text','class']]


# In[145]:


df = real.append(fake,ignore_index=True)


# In[146]:


df


# In[147]:


import string


# In[148]:


def clean_text(text:str):
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)
    return text


# In[149]:


# clean text
df['text'] = df['text'].apply(clean_text)


# In[150]:


# here we use vectorization 

import gensim


# In[151]:


y = df['class'].values


# In[152]:


X = [d.split() for d in df['text'].tolist()]


# In[153]:


type(X[0])


# In[154]:


print(X[0])


# In[156]:


DIM = 100
word_to_vector_model = gensim.models.Word2Vec(sentences=X,vector_size=DIM,window=10,min_count=1)


# In[157]:


len(word_to_vector_model.wv.index_to_key)


# In[158]:


word_to_vector_model.wv.most_similar('us')


# In[159]:


word_to_vector_model.wv.most_similar('hate')


# In[160]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)


# In[161]:


X = tokenizer.texts_to_sequences(X)


# In[162]:


tokenizer.word_index


# In[164]:


len(X)


# In[165]:


[len(x) for x in X]


# In[167]:


plt.hist([len(x) for x in X],bins=700)
plt.show();


# In[49]:


#Here we find News which have more than 1000 words

nos = np.array([len(x) for x in X])
len(nos[nos>1000])


# In[50]:


maxlen = 1000
X = pad_sequences(X,maxlen=maxlen)


# In[51]:


vocab_size = len(tokenizer.word_index) + 1


# In[52]:


vocab = tokenizer.word_index


# In[53]:


def get_weight_matrix(model):
    weight_matrix = np.zeros((vocab_size,DIM))
    
    for word,i in vocab.items():
        weight_matrix[i] = model.wv[word]
        
    return weight_matrix


# In[54]:


embedding_vectors = get_weight_matrix(w2v_model)


# In[55]:


model = Sequential()
model.add(Embedding(vocab_size,output_dim=DIM,weights = [embedding_vectors],input_length = maxlen , trainable=False))
model.add(LSTM(units=128))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])


# In[56]:


model.summary()


# In[57]:


x_train,x_test,y_train,y_test = train_test_split(X,y,random_state=10)


# In[58]:


model.fit(x_train,y_train,validation_split=0.3,epochs=1)


# In[59]:


y_pred = (model.predict(x_test)>= 0.5).astype(int)
accuracy_test = accuracy_score(y_test, y_pred)
accuracy_test


# In[60]:


df.head(10)


# In[168]:


df.iloc[6]['text']


# In[62]:


def predict_newstype(x:str):
    x = tokenizer.texts_to_sequences([x])
    x = pad_sequences(x,maxlen=maxlen)
    y_pred = (model.predict(x)>=0.5).astype(int)
    if y_pred==0:
        print('The news is fake')
    else:
        print('The news is real')


# In[169]:


predict_newstype('pakistan army pushed political role for militant linked groups the backing of a candidate in a by election last weekend in pakistan by a political party controlled by an islamist with a  10 million u s  bounty on his head was in line with a plan put forward by the military last year to mainstream militant groups  according to sources familiar with the proposal  the milli muslim league party loyal to hafiz saeed   who the united states and india accuse of masterminding the 2008 mumbai attacks that killed 166 people   won 5 percent of the votes in the contest for the seat vacated when prime minister nawaz sharif was removed from office by the supreme court in july  but the foray into politics by saeed s islamist charity appears to be following a blueprint that sharif himself rejected when the military proposed it in 2016  according to three government officials and a retired former general briefed on the discussions  none of the sources interviewed for this article could say for sure if the mml s founding was the direct result of the military s plan  which was not discussed in meetings after sharif put it on ice last year  the mml denies its political ambitions were engineered by the military  the official army spokesman did not comment after queries were sent to his office about the mainstreaming plan and what happened to it  pakistan s powerful military has long been accused of fostering militant groups as proxy fighters opposing neighboring arch enemy india  a charge the army denies  three government officials and close sharif confidants with knowledge of the discussions said the military s powerful inter services intelligence agency  isi  presented proposals for  mainstreaming  some militant groups in a meeting last year  they said that sharif had opposed the  mainstreaming  plan  which senior military figures and some analysts see as a way of steering ultra religious groups away from violent jihad   we have to separate those elements who are peaceful from the elements who are picking up weapons   said retired lieutenant general amjad shuaib  adding that such groups should be  helped out to create a political structure  to come into the mainstream  the plan   which shuaib told reuters was shared with him by the then head of the isi    said those who were willing  should be encouraged to come into the mainstream politics of the country   he added that in his capacity as a retired senior military officer he unofficially spoke to hafiz saaed and another alleged militant about the plan  and they were receptive   shuaib later said his comments in the interview were taken out of context and were part of a broader discussion about deradicalization strategies  writing in a local newspaper on wednesday he said the report  maliciously attributed some statements to me totally out of context  just to suit its own narrative   a spokesperson for reuters said   we stand by our reporting   saeed s religious charity launched the milli muslim league party within two weeks of the court ousting sharif over corruption allegations   yaqoob sheikh  the lahore candidate for milli muslim league  stood as an independent after the electoral commission said the party was not yet legally registered    but saeed s lieutenants  jud workers and mml officials ran his campaign and portraits of saeed adorn every poster promoting sheikh  who came in fourth place on sunday with sharif s wife taking the seat as expected  another islamist designated a terrorist by the united states  fazlur rehman khalil  has told reuters he too plans to soon form his own party to advocate strict islamic law   god willing  we will come into the mainstream   our country right now needs patriotic people   khalil said  vowing to turn pakistan into a state government by strict islamic law  saeed s charity and khalil s ansar ul umma organization are both seen by the united states as fronts for militant groups the army has been accused of sponsoring  the military denies any policy of encouraging radical groups  still  hundreds of mml supporters  waving posters of saeed and demanding his release from house arrest  chanted  long live hafiz saeed  long live the pakistan army   at political rallies during the run up to the by election   anyone who is india s friend is a traitor  a traitor   went another campaign slogan  a reference to sharif s attempts to improve relations with long time foe india that was a source of tension with the military  both saeed and khalil are proponents of a strict interpretation of islam and have a history of supporting violence   each man was reportedly a signatory to al qaeda leader osama bin laden s 1998 fatwa declaring war on the united states  they have since established religious groups that they say are unconnected to violence  though the united states maintains those groups are fronts for funneling money and fighters to militants targeting india  analyst khaled ahmed  who has researched saeed s jamaat ud dawa charity and its connections to the military  says the new political party is clearly an attempt by the generals to pursue an alternative to dismantling its militant proxies   one thing is the army wants these guys to survive   ahmed said   the other thing is that they want to also balance the politicians who are more and more inclined to normalize relations with india    the isi began pushing the political mainstreaming plan in 2016  according to retired general shuaib  a former director of the army s military intelligence wing that is separate from the isi  he said the proposal was shared with him in writing by the then isi chief  adding that he himself had spoken with khalil as well as saeed in an unofficial capacity about the plan   fazlur rehman khalil was very positive  hafiz saeed was very positive   shuaib said   my conversation with them was just to confirm those things which i had been told by the isi and other people   the isi s main press liaison did not respond to written requests for comment  saeed has been under house arrest since january at his house in the eastern city of lahore  the united states has offered a  10 million reward for information leading to his conviction over the mumbai attacks  then prime minister sharif  however  was strongly against the military s mainstreaming plan  according to shuaib and the three members of sharif s inner circle  including one who was in some of the tense meetings over the issue  sharif wanted to completely dismantle groups like jud  disagreement on what to do about anti india proxy fighters was a major source of rancor with the military  according to one of the close sharif confidants   in recent weeks several senior figures from the ruling pml n party have publicly implied that elements of the military   which has run pakistan for almost half its modern history and previously ousted sharif in a 1999 coup   had a hand in the court ouster of sharif  a charge both the army and the court reject  a representative of the pml n  which last month replaced him as prime minister with close ally shahid khaqi abbasi  said the party was  not aware  of any mainstreaming plan being brought to the table ')


# In[ ]:




