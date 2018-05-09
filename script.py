# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
#os.chdir('C:\\Users\\ajadhav\\Desktop\\SongHottness')
train  = pd.read_csv("train.csv")
artist = pd.read_csv("artists.csv")
test = pd.read_csv("test.csv")
train = train[train.year!=1]
train = train.merge(artist, how = 'inner', on = 'artist_id')
test = test.merge(artist, how = 'inner', on = 'artist_id')
del artist

train.song_hotttnesss[train.song_hotttnesss>train.song_hotttnesss.mean()] = 1
train.song_hotttnesss[train.song_hotttnesss<train.song_hotttnesss.mean()] = 0

train.year = (train.year//10)*10
test.year = (test.year//10)*10

CategoricalFeatures = train[['artist_id','title','audio_md5']]

from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer
vectorizer = HashingVectorizer(n_features=750)
#vectorizer = TfidfVectorizer(min_df  = 0.0002)

TfidfVectorizerObject = vectorizer.fit(pd.concat([train.title ,test.title]))
CountVectorizerTrainData = TfidfVectorizerObject.transform(train["title"])
CountVectorizerTestData = TfidfVectorizerObject.transform(test["title"])

DropFeatures = ['song_id','artist_id','title','audio_md5','analysis_sample_rate',
                'key_confidence','audio_md5','year','end_of_fade_in','duration',
                'time_signature_confidence','artist_latitude',
                'artist_longitude']

trainSongId = train[['song_id']]
train = train.drop(DropFeatures, axis=1)
song_id = test['song_id']
test = test.drop(DropFeatures, axis=1)
train = pd.concat([train, pd.DataFrame(CountVectorizerTrainData.toarray())], axis=1)
test = pd.concat([test, pd.DataFrame(CountVectorizerTestData.toarray())], axis=1)
X_train, X_test, y_train, y_test = train_test_split(train.drop('song_hotttnesss',axis=1),train['song_hotttnesss'], test_size=0.1, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

TestPred = model.predict(X_test)

print('Score on validation set is',print(classification_report(y_test,TestPred)))

song_hotttnesss = model.predict_proba(test)
song_hotttnesss = song_hotttnesss[:,:1]
submission = pd.DataFrame({'song_id': song_id})
submission['song_hotttnesss'] = song_hotttnesss
#submission= submission[['song_id','song_hotttnesss']]
submission.to_csv("submission.csv", index= False)

