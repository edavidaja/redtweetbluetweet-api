import json
import pickle
import nltk
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

tokenizer = nltk.casual.TweetTokenizer(preserve_case=False, reduce_len=True)

data = pd.read_feather("data/tweets.feather")

with open("data/demonyms.json") as f:
    demonyms = json.load(f)

reg = Pipeline(
    steps=[
        ("tfidf", TfidfVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 2))),
        ("classifier", MultinomialNB()),
    ]
)

X = data["full_text"].values.tolist()
y = data["party_code"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

grid_parameters = {"tfidf__min_df": (0.1, 1)}

grid_search = GridSearchCV(reg, grid_parameters, cv=5, n_jobs=-1, verbose=1)

grid_search.fit(X_test, y_test)

with open("model/model.pkl", "wb") as f:
    pickle.dump(grid_search.best_estimator_, f)
