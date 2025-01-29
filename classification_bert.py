# %%
import torch

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from pathlib import Path

Path.mkdir(Path("data/embeddings"), exist_ok=True, parents=True)


def get_embeds(texts):
    with torch.no_grad():
        return [model_BERT.encode(c) if not pd.isna(c) else np.NaN for c in texts]


from sentence_transformers import SentenceTransformer

model_names = [
    "dominguesm/legal-bert-base-cased-ptbr",
    "neuralmind/bert-base-portuguese-cased",
]

for model in model_names:
    if model == "dominguesm/legal-bert-base-cased-ptbr":
        model_name = "legal_bert"
    else:
        model_name = "bert"

    model_BERT = SentenceTransformer(model)

    df = pd.read_csv("data/preprocessed/texts_without_ANVISA_ANEEL_mask.csv")
    df = df.dropna(subset=["text"])

    # get embeddings
    embeddings = get_embeds(df["text"])
    X = np.array(embeddings)

    np.save(f"data/embeddings/embeddings_{model_name}_training.npy", X)

    X = np.load(f"data/embeddings/embeddings_{model_name}_training.npy")
    y = df["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    df_validation = pd.read_csv("data/preprocessed/texts_ANVISA_ANEEL_mask.csv")
    df_validation = df_validation.dropna(subset=["text"])


    df_validation = df_validation.reset_index(drop=True)

    embeddings_validation = get_embeds(df_validation["text"])

    X_validation = np.array(embeddings_validation)

    np.save(f"data/embeddings/embeddings_{model_name}_validation.npy", X_validation)

    X_validation = np.load(f"data/embeddings/embeddings_{model_name}_validation.npy")

    embeddings = np.concatenate([X, X_validation], axis=0)
    np.save(f"data/embeddings/embeddings_{model_name}.npy", embeddings)
    
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
        KNeighborsClassifier(
            n_neighbors=3,
        ),
        DecisionTreeClassifier(max_depth=4),
        RandomForestClassifier(max_depth=3),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        SVC(kernel="linear"),
        SVC(kernel="rbf"),
        SVC(kernel="sigmoid"),
        LogisticRegression(max_iter=250),
    ]

    models = zip(names, classifiers)

    results = []
    for name, model in models:

        print(name)

        model = model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        print(classification_report(y_test, predictions))

        df_conc = df_validation.query(
            "labels == 'concreta'"
        )["text"]
        aneel_conc_indices = df_conc.index
        X_aneel_conc = X_validation[aneel_conc_indices]

        y_abs = model.predict(X_aneel_conc.tolist())
        acuracia = accuracy_score(y_abs, ["concreta"] * len(y_abs))
        print(
            f"Acurácia Concretas: {acuracia*100:.2f}% ({Counter(y_abs)['concreta']} de {len(y_abs)})"
        )

        df_abs = df_validation.query(
            "labels == 'abstrata'"
        )["text"]
        aneel_conc_indices = df_abs.index
        X_aneel_conc = X_validation[aneel_conc_indices]

        y_abs = model.predict(X_aneel_conc.tolist())
        acuracia = accuracy_score(y_abs, ["abstrata"] * len(y_abs))
        print(
            f"Acurácia abstratas: {acuracia*100:.2f}% ({Counter(y_abs)['abstrata']} de {len(y_abs)})"
        )




