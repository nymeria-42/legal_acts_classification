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
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import pandas as pd

from collections import Counter

from shap import Explainer
from shap.plots import beeswarm


df = pd.read_csv("data/preprocessed/texts_without_ANVISA_ANEEL.csv")
df.dropna(subset=["text"], inplace=True)

labels = df["labels"]
texts = df["text"]

X = df["text"]
y = df['labels']


X_train, X_test, y_train, y_test, texts_train, texts_test = train_test_split(
    X, y, df['text'], test_size=0.3, random_state=42
)


vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)



df_validacao = pd.read_csv("data/preprocessed/texts_ANVISA_ANEEL.csv")
df_validacao.dropna(subset=["text"], inplace=True)


texts_conc = df_validacao.query("labels == 'concreta'")["text"]
texts_conc = vectorizer.transform(texts_conc).toarray()

texts_abs = df_validacao.query("labels == 'abstrata'")["text"]
texts_abs = vectorizer.transform(texts_abs).toarray()


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

    print(name)

    model = model.fit(X_train.toarray(), y_train)
    predictions = model.predict(X_test.toarray())
    print(classification_report(y_test, predictions))


    try:
        explainer = Explainer(model, feature_names=vectorizer.get_feature_names_out())
        sv = explainer(X_train.toarray())
        
        beeswarm(sv[:,:,0], max_display=10)
    except:
        pass

    y_conc = model.predict(texts_conc)
    print("Concretas")
    acuracia = accuracy_score(y_conc, ['concreta']*len(y_conc))
    print(f"Acurácia: {acuracia*100:.2f}% ({Counter(y_conc)['concreta']} de {len(y_conc)})")

    y_abs = model.predict(texts_abs)
    print("Abstratas")
    acuracia = accuracy_score(y_abs, ['abstrata']*len(y_abs))
    print(f"Acurácia: {acuracia*100:.2f}% ({Counter(y_abs)['abstrata']} de {len(y_abs)})")

    # calculate f1 score
    y_true_general = ['concreta']*len(y_conc) + ['abstrata']*len(y_abs)
    y_pred_general = list(y_conc) + list(y_abs)

    f1_general = f1_score(y_true_general, y_pred_general, average="macro")
    print(f"F1 Score: {f1_general}")
