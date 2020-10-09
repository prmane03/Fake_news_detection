
"""
### Fake News Classifier
Dataset:  https://www.kaggle.com/c/fake-news/data#
"""

import pandas as pd

# Reading data
df=pd.read_csv('fake-news/train.csv')

## Get the Independent Features
X=df.drop('label',axis=1)

## Get the Dependent features
y=df['label']

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer

df=df.dropna()

messages=df.copy()

# %%
messages.reset_index(inplace=True)

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['text'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# %%
## TFidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,3))
X=tfidf_v.fit_transform(corpus).toarray()


## Divide the dataset into Train and Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

#tfidf_v.get_feature_names()[:20]
#tfidf_v.get_params()

count_df = pd.DataFrame(X_train, columns=tfidf_v.get_feature_names())

# %%
count_df.head()

""""# %%
import matplotlib.pyplot as plt

# %%
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
#
    See full source and example: 
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
#  
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
"""
# %%
"""
### MultinomialNB Algorithm
"""

# %%

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()

# %%
from sklearn import metrics
import numpy as np
import itertools

# %%

classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)
#cm = metrics.confusion_matrix(y_test, pred)
#plot_confusion_matrix(cm, classes=['FAKE', 'REAL'])

# %%
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
score = metrics.accuracy_score(y_test, pred)
print("accuracy:   %0.3f" % score)

# %%
#y_train.shape

