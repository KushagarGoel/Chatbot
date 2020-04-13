# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 21:19:48 2020

@author: kushagar
"""

import nltk 
import numpy as np 
import string 

f=open('covid_cb.txt','r',errors = 'ignore')
raw=f.read().lower()
sent = nltk.sent_tokenize(raw)# converts to list of sentences

lemmer = nltk.stem.WordNetLemmatizer() #WordNet is a semantically-oriented dictionary of English included in NLTK. 
def LemTokens(tokens):    
    return [lemmer.lemmatize(token) for token in tokens] 
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation) 
def LemNormalize(text):    
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):    
    robo_response=''    
    sent.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')    
    tfidf = TfidfVec.fit_transform(sent)  
    vals = cosine_similarity(tfidf[-1], tfidf)  
    idx = vals.argsort()[0][-2]     
    if(idx==len(sent)-2):        
        robo_response=robo_response+"Sorry, but could not understand, please try another one"        
        return robo_response    
    else:        
        robo_response = sent[idx]        
        return robo_response



termi = ['bye','exit','quit','thanks','thanku','thank you']
greets = ["hello", "hi", "greetings","hey"]

f=True 
print("Covid assist:  I will answer your queries about Coronavirus. If you want to exit, type Bye!") 
while(f==True):    
    user_response = input().lower()    
    if(user_response not in termi):        
        if(user_response in greets):                
            print("Covid assist: I would be glad to help you")            
        else:                
            print("Covid assist: ",end="")                
            print(response(user_response))                
            sent.remove(user_response)    
    else:        
        f=False        
        print("Covid assist: Bye! take care..")