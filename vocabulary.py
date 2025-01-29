
#%%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# pathlib
from pathlib import Path

Path('data/vocabulary').mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/preprocessed/texts_without_ANVISA_ANEEL.csv")

df_anvisa_aneel = pd.read_csv("data/preprocessed/texts_ANVISA_ANEEL.csv")
# rename column agency_abbrev to agency
df_anvisa_aneel.rename(columns={"agency_abbrev": "agency"}, inplace=True)
df = pd.concat([df, df_anvisa_aneel])
df.dropna(subset=["text"], inplace=True)
df.to_csv("data/dados.csv")

#%%
# TFIDF - labels
grouped_texts = df.groupby('labels')['text'].apply(' '.join)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_grouped = vectorizer.fit_transform(grouped_texts)

# Extract vocabulary and scores
vocabulary = vectorizer.get_feature_names_out()

tfidf_matrix = X_grouped.toarray()

df_tfidf = pd.DataFrame(tfidf_matrix, columns=vocabulary)
df_tfidf["label"] = grouped_texts.index
df_tfidf = df_tfidf.set_index("label")
df_tfidf.to_csv("data/vocabulary/tfidf_matrix_label.csv")

# TFIDF - act type
grouped_texts = df.groupby('publication_type')['text'].apply(' '.join)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_grouped = vectorizer.fit_transform(grouped_texts)

vocabulary = vectorizer.get_feature_names_out()

bow_matrix = X_grouped.toarray()

df_bow = pd.DataFrame(bow_matrix, columns=vocabulary)

df_bow["type"] = grouped_texts.index
df_bow = df_bow.set_index("type")
df_bow.to_csv("data/vocabulary/tfidf_matrix_type.csv")

# TFIDF - agency
grouped_texts = df.groupby('agency')['text'].apply(' '.join)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_grouped = vectorizer.fit_transform(grouped_texts)

vocabulary = vectorizer.get_feature_names_out()

tfidf_matrix = X_grouped.toarray()

df_tfidf = pd.DataFrame(tfidf_matrix, columns=vocabulary)
df_tfidf["agency"] = grouped_texts.index
df_tfidf = df_tfidf.set_index("agency")
df_tfidf.to_csv("data/vocabulary/tfidf_matrix_agency.csv")

# TFIDF - all
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))


X_grouped = vectorizer.fit_transform(df["text"])

vocabulary = vectorizer.get_feature_names_out()

tfidf_matrix = X_grouped.toarray()

df_tfidf = pd.DataFrame(tfidf_matrix, columns=vocabulary)
df_tfidf.to_csv("data/vocabulary/tfidf_matrix.csv")

# BOW - agency
grouped_texts = df.groupby('agency')['text'].apply(' '.join)

df.groupby('agency').size().to_csv("data/acts_per_agency.csv")

vectorizer = CountVectorizer(max_features=None, ngram_range=(1, 1), binary=True)

X_grouped = vectorizer.fit_transform(grouped_texts)

vocabulary = vectorizer.get_feature_names_out()

bow_matrix = X_grouped.toarray()

df_bow = pd.DataFrame(bow_matrix, columns=vocabulary)

df_bow["agency"] = grouped_texts.index
df_bow = df_bow.set_index("agency")
df_bow.to_csv("data/vocabulary/bow_matrix_agency.csv")
#%%
# BOW - act type
grouped_texts = df.groupby('publication_type')['text'].apply(' '.join)

df.groupby('publication_type').size().to_csv("data/acts_per_type.csv")

vectorizer = CountVectorizer(max_features=None, ngram_range=(1, 1), binary=True)

X_grouped = vectorizer.fit_transform(grouped_texts)

vocabulary = vectorizer.get_feature_names_out()

bow_matrix = X_grouped.toarray()

df_bow = pd.DataFrame(bow_matrix, columns=vocabulary)

df_bow["type"] = grouped_texts.index
df_bow = df_bow.set_index("type")
df_bow.to_csv("data/vocabulary/bow_matrix_type.csv")

# BOW - labels
grouped_texts = df.groupby('labels')['text'].apply(' '.join)

df.groupby('labels').size().to_csv("data/acts_per_label.csv")

vectorizer = CountVectorizer(max_features=None, ngram_range=(1, 1), binary=True)

X_grouped = vectorizer.fit_transform(grouped_texts)

vocabulary = vectorizer.get_feature_names_out()

bow_matrix = X_grouped.toarray()

df_bow = pd.DataFrame(bow_matrix, columns=vocabulary)
df_bow["label"] = grouped_texts.index
df_bow = df_bow.set_index("label")
df_bow.to_csv("data/vocabulary/bow_matrix_label.csv")

# BOW - all
vectorizer = CountVectorizer(max_features=None, ngram_range=(1, 1), binary=True)


X_grouped = vectorizer.fit_transform(df["text"])

vocabulary = vectorizer.get_feature_names_out()

bow_matrix = X_grouped.toarray()

df_tfidf = pd.DataFrame(bow_matrix, columns=vocabulary)
df_tfidf.to_csv("data/vocabulary/bow_matrix.csv")

# TFIDF - within agency

grouped_texts = df.groupby('agency')['text'].apply(list)

all_texts = df['text'].tolist()

# 1. Global TF-IDF calculation
print("=== Global TF-IDF Calculation ===")
global_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 1))
X_corpus = global_vectorizer.fit_transform(all_texts)

global_vocabulary = global_vectorizer.get_feature_names_out()
global_tfidf_matrix = X_corpus.toarray()

global_agency_summaries = {}

start_idx = 0
for agency, texts in grouped_texts.items():
    end_idx = start_idx + len(texts)
    agency_matrix = global_tfidf_matrix[start_idx:end_idx]

    agency_scores = agency_matrix.sum(axis=0)
    word_scores = list(zip(global_vocabulary, agency_scores))
    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:10]  # Top 10 words

    global_agency_summaries[agency] = sorted_words

    start_idx = end_idx

for agency, words in global_agency_summaries.items():
    print(f"\nAgency: {agency} (Global TF-IDF)")
    for word, score in words:
        print(f"  {word}: {score:.4f}")

# 2. Agency-specific TF-IDF calculation
print("\n=== Agency-Specific TF-IDF Calculation ===")
specific_agency_summaries = {}

for agency, texts in grouped_texts.items():
    agency_vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 1))
    X_agency = agency_vectorizer.fit_transform(texts)

    agency_vocabulary = agency_vectorizer.get_feature_names_out()
    agency_tfidf_matrix = X_agency.toarray()

    agency_scores = agency_tfidf_matrix.sum(axis=0)
    word_scores = list(zip(agency_vocabulary, agency_scores))
    sorted_words = sorted(word_scores, key=lambda x: x[1], reverse=True)[:10]  # Top 10 words

    specific_agency_summaries[agency] = sorted_words

for agency, words in specific_agency_summaries.items():
    print(f"\nAgency: {agency} (Agency-Specific TF-IDF)")
    for word, score in words:
        print(f"  {word}: {score:.4f}")
