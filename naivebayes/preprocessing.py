import os
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from time import time
import random
import datetime
import numpy as np
from dict.loadsDict import stopwords
import re


currentpath = os.path.dirname(__file__)
resourcespath = "{}/resources".format(currentpath)

resourcefiles = []

for (dirpath, dirnames, filenames) in os.walk(resourcespath):
    resourcefiles.extend(filenames)

resources = []
print("Read resources ...")
for index,resourcefile in enumerate(resourcefiles):
    print("({}/{})".format(index+1,len(resourcefiles)))
    print("Read resources process {}".format(resourcefile))
    resources.append(pd.read_excel("{}/{}".format(resourcespath,resourcefile)))
    print("Read resources success {}".format(resourcefile))
resources = pd.concat(resources)

tweets = resources.loc[:,['tweet']].to_numpy().flatten()
labels = resources.loc[:,['label']].to_numpy().flatten()

stemmer = StemmerFactory().create_stemmer()

print("Stemming Processs ...")
for index,tweet in enumerate(tweets):
    print("Stemming ({}/{})".format(index+1,len(tweets)))
    tweet = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', tweet, flags=re.MULTILINE)
    tweets[index] = stemmer.stem(tweet)

features_train, features_test, labels_train, labels_test = train_test_split(tweets,labels,test_size=0.3,train_size=0.7)
vectorizer = TfidfVectorizer(stop_words=stopwords)
features_train  = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test)

t0 = time()
model = GaussianNB()
model.fit(features_train.todense(), labels_train)
print(f"\nTraining time: {round(time()-t0, 3)}s")
t0 = time()
score_train = model.score(features_train.todense(), labels_train)
print(f"Prediction time (train): {round(time()-t0, 3)}s")
t0 = time()
score_test = model.score(features_test.todense(), labels_test)
print(f"Prediction time (test): {round(time()-t0, 3)}s")
print("\nTrain set score:", score_train)
print("Test set score:", score_test)

resource_feature = vectorizer.transform(tweets)
predicts = np.zeros(resource_feature.shape[0])
for index,item in enumerate(predicts):
    predicts[index] = model.predict(resource_feature[index].todense())[0]

resources['predict'] = predicts
resources['tweet_stemmed'] = tweets

resources.to_excel("{}/results/{}.xlsx".format(currentpath,datetime.date.today()),index=False)