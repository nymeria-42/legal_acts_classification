# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd

from collections import Counter

from shap import Explainer
from shap.plots import beeswarm


df = pd.read_csv("data/preprocessed/texts_ANVISA_ANEEL.csv")
df.dropna(subset=["text"], inplace=True)
df["publication_date"] = pd.to_datetime(df["publication_date"])
df = df.sort_values("publication_date")
df.reset_index(drop=True, inplace=True)


labels = df["labels"]
texts = df["text"]

X = df["text"]
y = df['labels']

# leave 2022 for validation
X_train_val = df[df["publication_date"] < "2022-01-01"]["text"]
X_test = df[df["publication_date"] >= "2022-01-01"]["text"]
y_train_val = df[df["publication_date"] < "2022-01-01"]["labels"]
y_test = df[df["publication_date"] >= "2022-01-01"]["labels"]


X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val,  test_size=0.3, random_state=42 
)


vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(X_train)

X_val = vectorizer.transform(X_val)

X_test = vectorizer.transform(X_test)

scoring = 'accuracy'

names = [
          "Nearest Neighbors", 
         "Decision Tree", 
         "Random Forest",
         "Neural Net", 
         "AdaBoost", 
         "Naive Bayes", 
         "SVM Linear", 
         "SVM RBF", 
         "SVM Sigmoid",
         "Logistic Regression",
         ]

classifiers = [
    KNeighborsClassifier(n_neighbors=3, ),
    DecisionTreeClassifier(max_depth=4),
    RandomForestClassifier(max_depth=3),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    SVC(kernel='linear'),
    SVC(kernel='rbf'),
    SVC(kernel='sigmoid'),
    LogisticRegression(max_iter=250),
]


models = zip(names, classifiers)

results = []
for name, model in models:

    print("=======================")
    print(name)
    model = model.fit(X_train.toarray(), y_train)
    predictions = model.predict(X_val.toarray())
    print(classification_report(y_val, predictions))

    test_predictions = model.predict(X_test.toarray())
    print("         ---- validation -----")
    print(classification_report(y_test, test_predictions))

    try:
        explainer = Explainer(model, feature_names=vectorizer.get_feature_names_out())
        sv = explainer(X_train.toarray())
        
        beeswarm(sv[:,:,0], max_display=10)
    except:
        pass


    