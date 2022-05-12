import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline


def run(train, test):

    # define the model
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), lowercase=False, analyzer='word', min_df=5)),
        ('clf', OneVsRestClassifier(LinearSVC(dual=False, max_iter=100000), n_jobs=-1))])
    text_clf.fit(train['text'], train['type'])

    # get predicted results
    predict = text_clf.predict(test['text'])

    # write to .csv
    sub = pd.DataFrame({'index': range(0, len(predict)), 'type': predict})
    sub.to_csv('class_answer.csv', index=False)
