# Auto-converted from notebook

# %% [cell 1: code]
!pip install -q deplacy
!pip install -q lexicalrichness

# %% [cell 2: code]
import numpy as np
import pandas as pd
import random
import glob
import librosa
import os
import string
from copy import deepcopy
from sklearn.feature_extraction.text import CountVectorizer #convert text comment into a numeric vector
from sklearn.feature_extraction.text import TfidfTransformer #use TF IDF transformer to change text vector created by count vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import SVC# Support Vector Machine
from sklearn.pipeline import Pipeline #pipeline to implement steps in series
from lightgbm import LGBMClassifier
from sklearn.ensemble import *
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
# import preprocessor as p
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
# import gensim
from sklearn.model_selection import train_test_split
import textblob
nltk.download('averaged_perceptron_tagger')
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import re
import spacy
# specify GPU
import torch
device = torch.device("cuda")
import spacy
import en_core_web_sm
from sklearn.feature_extraction.text import CountVectorizer
from lexicalrichness import LexicalRichness
from collections import Counter
from string import punctuation
import math
import deplacy
from copy import deepcopy
from collections import Counter

# %% [cell 3: markdown]
# # Load Dataset

# %% [cell 4: code]
path = '/workspace/'

# %% [cell 5: code]
pd.set_option('display.max_rows', 700)

# %% [cell 6: code]
DATA_PATH = path + 'Data/'

FILE_PATH = DATA_PATH + './train_df_story_recall_cinderella.csv' # 4-1 were tested.

class_label = ["Control_group", "Ad_Alzhiemer"]

# %% [cell 7: code]
df = pd.read_csv(FILE_PATH)

# %% [cell 8: markdown]
# # Data Preprocessing

# %% [cell 9: code]
if ('transcription' in df.columns) and ('text' not in df.columns):
    df = df.rename(columns={'transcription':'text'})
df.shape

# %% [cell 10: code]
def preprocesssing(df_text):
    # remove punctuation marks
    df_text = df_text.rename(columns={'Text': 'text'})
    df_text['clean_text'] = df_text['text'].apply(lambda x: re.sub(r'http\S+', '', x))

    punctuation = '!"#$%&()*+-/:;<=>?@[\\]^_`{|}~'

    df_text['clean_text'] = df_text['clean_text'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))
    # test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ''.join(ch for ch in x if ch not in set(punctuation)))

    # convert text to lowercase
    df_text['clean_text'] = df_text['clean_text'].str.lower()
    # test['clean_tweet'] = test['clean_tweet'].str.lower()

    # remove numbers
    df_text['clean_text'] = df_text['clean_text'].str.replace("[0-9]", " ")
    # test['clean_tweet'] = test['clean_tweet'].str.replace("[0-9]", " ")

    # remove whitespaces
    df_text['clean_text'] = df_text['clean_text'].apply(lambda x:' '.join(x.split()))
    # test['clean_tweet'] = test['clean_tweet'].apply(lambda x: ' '.join(x.split()))
    return df_text

# %% [cell 11: code]
df_text = df.copy()
df_text = preprocesssing(deepcopy(df_text))

# %% [cell 12: markdown]
# # POS

# %% [cell 13: code]
# load en_core_web_sm of English for vocabluary, syntax & entities
nlp = en_core_web_sm.load()
#  "nlp" Objectis used to create documents with linguistic annotations.
df_text["POS"] = df_text["clean_text"].apply(lambda x:" ".join([word.pos_ for word in nlp(x)]))

vectorizer = CountVectorizer()
X_v = vectorizer.fit_transform(df_text["POS"].tolist())
df_pos = pd.DataFrame(data=X_v.toarray(), columns = [pos.upper() for pos in vectorizer.get_feature_names_out()])

df_text_pos = pd.concat([df_text, df_pos], axis=1)
df_text_pos.head(1)

# %% [cell 14: markdown]
# # TAG

# %% [cell 15: code]
df_text_pos["TAG"] = df_text["clean_text"].apply(lambda x:" ".join([word.tag_ for word in nlp(x)]))

vectorizer1 = CountVectorizer()
X_tag = vectorizer1.fit_transform(df_text_pos["TAG"].tolist())
df_tag = pd.DataFrame(data=X_tag.toarray(), columns = [tag.upper() for tag in vectorizer1.get_feature_names_out()])

df_t_p_t = pd.concat([df_text_pos, df_tag], axis=1)
df_t_p_t.head(2)

# %% [cell 16: markdown]
# # Some of Linguistic Features

# %% [cell 17: markdown]
# ## Content Density

# %% [cell 18: code]
df_t_p_t["Content Density"] = (df_t_p_t["ADJ"] + df_t_p_t["ADV"] + df_t_p_t["VERB"] + df_t_p_t["NOUN"]) / df_t_p_t.loc[:, "ADJ":"VERB"].drop(["ADJ", "ADV", "VERB", "NOUN", "PUNCT"], axis=1).sum(axis=1)
df_t_p_t.head(2)

# %% [cell 19: markdown]
# ## Part-of-Speech rate

# %% [cell 20: code]
pos = ["ADJ", "ADV", "DET", "CONJ", "INTJ", "NOUN", "NUM", "IN", "PRON", "VERB"]
sum_pos = []
for i in range(len(pos)):
    if pos[i] == "CONJ":
        Seri_CONJ = df_t_p_t["CCONJ"] + df_t_p_t["SCONJ"]
        sum_pos.append(Seri_CONJ / Seri_CONJ.sum())
    else:
        sum_pos.append(df_t_p_t[pos[i]] / df_t_p_t[pos[i]].sum())

df_t_p_t["Part-of-Speech rate"] = sum(sum_pos) / 10
df_t_p_t.head(2)

# %% [cell 21: markdown]
# ## Reference Rate to Reality

# %% [cell 22: code]
df_t_p_t["Reference Rate to Reality"] = df_t_p_t["NOUN"] / df_t_p_t["VERB"]
df_t_p_t.head(2)

# %% [cell 23: markdown]
# ## Relative pronouns rate

# %% [cell 24: code]
df_t_p_t["Relative pronouns rate"] = (df_t_p_t["WDT"] + df_t_p_t["WP"]) / (df_t_p_t["WDT"] + df_t_p_t["WP"]).sum()

# %% [cell 25: markdown]
# ## Negative adverbs rate

# %% [cell 26: code]
def negative_adverb_count(x):
    cnt = 0
    sid = SentimentIntensityAnalyzer()
    for word in nlp(x):
        if word.pos_ == "ADV":
            if sid.polarity_scores(str(word))['neg'] > sid.polarity_scores(str(word))['pos']:
                cnt += 1
    return cnt

# %% [cell 27: code]
Seri_Neg_adv = df_t_p_t["clean_text"].apply(negative_adverb_count)
df_t_p_t["Negative adverbs rate"] = Seri_Neg_adv / Seri_Neg_adv.sum()
df_t_p_t.head(1)

# %% [cell 28: markdown]
# ## Filler words

# %% [cell 29: code]
filler_list = ["uh", "um", "hmm", "mhm", "huh"]
def Filler_word_count(x):
    cnt = 0
    for word in nlp(x):
        if str(word.text) in filler_list:
            cnt += 1
    return cnt

# %% [cell 30: code]
df_t_p_t["Filler words"] = df_t_p_t["clean_text"].apply(Filler_word_count)
df_t_p_t.head(1)

# %% [cell 31: markdown]
# ## Action Verbs rate

# %% [cell 32: code]
df_action_verb = pd.read_excel(path + "Data/Action_verbs/action_verbs.xlsx", header=None)
action_verb = df_action_verb.astype(str).to_numpy().reshape(-1)
action_verb = np.unique(action_verb)
action_verb = np.delete(action_verb, np.where(action_verb=='nan')[0][0])
def action_verb_count(x):
    cnt = 0
    for word in nlp(x):
        if str(word.lemma_) in action_verb:
            cnt += 1
    return cnt

# %% [cell 33: code]
df_t_p_t["action_verb"] = df_t_p_t["clean_text"].apply(action_verb_count)
df_t_p_t.head(1)

# %% [cell 34: markdown]
# ## Lexical frequency

# %% [cell 35: code]
real_pos_list = ['VERB', 'PROPN', 'NOUN', 'ADV', 'ADJ']
sum_real = []
for real_pos in real_pos_list:
    sum_real.append(df_t_p_t[real_pos])
df_t_p_t["Lexical frequency"] = np.log2(sum(sum_real))
df_t_p_t.head(1)

# %% [cell 36: markdown]
# ## Definite articles

# %% [cell 37: code]
def the_count(x):
    cnt = 0
    for word in nlp(x):
        if str(word.text) == "the":
            cnt += 1
    return cnt

# %% [cell 38: code]
df_t_p_t["Definite articles"] = df_t_p_t["clean_text"].apply(the_count) / df_t_p_t.loc[:, "ADJ":"VERB"].drop("PUNCT", axis=1).sum(axis=1)

# %% [cell 39: markdown]
# ## Indefinites articles

# %% [cell 40: code]
a_an_list = ["a", "an"]
def a_an_count(x):
    cnt = 0
    for word in nlp(x):
        if str(word.text) in a_an_list:
            cnt += 1
    return cnt

# %% [cell 41: code]
df_t_p_t["Indefinites articles"] = df_t_p_t["clean_text"].apply(a_an_count) / df_t_p_t.loc[:, "ADJ":"VERB"].drop("PUNCT", axis=1).sum(axis=1)
df_t_p_t.head(1)

# %% [cell 42: markdown]
# ## Pronouns

# %% [cell 43: code]
df_t_p_t["Pronouns"] = df_t_p_t["PRON"] / df_t_p_t.loc[:, "ADJ":"VERB"].drop("PUNCT", axis=1).sum(axis=1)

# %% [cell 44: markdown]
# ## Nouns

# %% [cell 45: code]
df_t_p_t["Nouns"] = df_t_p_t["NOUN"] / df_t_p_t.loc[:, "ADJ":"VERB"].drop("PUNCT", axis=1).sum(axis=1)

# %% [cell 46: markdown]
# ## Verbs

# %% [cell 47: code]
df_t_p_t["Verbs"] = df_t_p_t["VERB"] / df_t_p_t.loc[:, "ADJ":"VERB"].drop("PUNCT", axis=1).sum(axis=1)

# %% [cell 48: markdown]
# ## Determiners

# %% [cell 49: code]
df_t_p_t["Determiners"] = df_t_p_t["DET"] / df_t_p_t.loc[:, "ADJ":"VERB"].drop("PUNCT", axis=1).sum(axis=1)

# %% [cell 50: markdown]
# ## Content words

# %% [cell 51: code]
def stop_word_spacy(x):
    cnt = 0
    for word in nlp(x):
        if word.is_stop == False:
            cnt += 1
    return cnt

# %% [cell 52: code]
df_t_p_t["Content words"] = df_t_p_t["clean_text"].apply(stop_word_spacy) / df_t_p_t.loc[:, "ADJ":"VERB"].drop("PUNCT", axis=1).sum(axis=1)

# %% [cell 53: markdown]
# ## Consecutive repeated clauses

# %% [cell 54: code]
def find_repeated_clauses(text):
    match = re.findall(r'((\b.+?\b)(?:\s\2)+)', text)
    repeated_clauses = [(m[1], int((len(m[0]) + 1) / (len(m[1]) + 1))) for m in match]
    count_list = [count for repeated_clause, count in repeated_clauses]
    return sum(count_list)

# %% [cell 55: code]
df_t_p_t["Consecutive repeated clauses"] = df_t_p_t['clean_text'].apply(find_repeated_clauses)
df_t_p_t.head(1)

# %% [cell 56: markdown]
# ## Lexical Richness: Type-Token Ratio,  and

# %% [cell 57: code]
# type-token ratio (TTR)
# root type-token ratio (RTTR)
# corrected type-token ratio (CTTR)
# mean segmental type-token ratio (MSTTR)
# moving average type-token ratio (MATTR)
df_t_p_t['TTR'] = df_t_p_t['clean_text'].apply(lambda x:LexicalRichness(x).ttr)
df_t_p_t['RTTR'] = df_t_p_t['clean_text'].apply(lambda x:LexicalRichness(x).rttr)
df_t_p_t['CTTR'] = df_t_p_t['clean_text'].apply(lambda x:LexicalRichness(x).cttr)
df_t_p_t.head(1)

# %% [cell 58: markdown]
# ## Total number of words

# %% [cell 59: code]
df_t_p_t['Word count'] = df_t_p_t['clean_text'].apply(lambda x:LexicalRichness(x).words)

# %% [cell 60: markdown]
# ## Total number of unique words

# %% [cell 61: code]
df_t_p_t['Unique Word count'] = df_t_p_t['clean_text'].apply(lambda x:LexicalRichness(x).terms)

# %% [cell 62: markdown]
# ## Ratio of total number of unique words to total number of words

# %% [cell 63: code]
df_t_p_t['Ratio unique word count to total word count'] = df_t_p_t['Unique Word count'] / df_t_p_t['Word count']

# %% [cell 64: markdown]
# ## W - Brunet’s Index

# %% [cell 65: code]
df_t_p_t['Brunet’s Index'] = df_t_p_t['Word count'] ** (df_t_p_t['Unique Word count'] ** (-0.165))

# %% [cell 66: markdown]
# ## R - Honor’s Statistic

# %% [cell 67: code]
def honore(text):
    try:
        words = [word.strip(punctuation) for word in text.lower().split()]
        N = len(words)
        types = set(words)
        V = len(types)
        V1 = sum(1 for w in words if words.count(w) == 1)
        H = 100 * math.log(N) / (1 - V1/V)
    except ZeroDivisionError:
        H = np.nan
    return H

# %% [cell 68: code]
df_t_p_t['Honor’s Statistic'] = df_t_p_t['clean_text'].apply(honore)
df_t_p_t['Honor’s Statistic'] = df_t_p_t['Honor’s Statistic'].fillna(df_t_p_t['Honor’s Statistic'].max())

# %% [cell 69: markdown]
# ## Measure of Textual Lexical Diversity (MTLD)

# %% [cell 70: code]
df_t_p_t['Measure of Textual Lexical Diversity'] = df_t_p_t['clean_text'].apply(lambda x:LexicalRichness(x).mtld(threshold=0.72))

# %% [cell 71: markdown]
# ## Hypergeometric distribution diversity (HD-D) measure

# %% [cell 72: code]
df_t_p_t['Hypergeometric Distribution Diversity'] = df_t_p_t['clean_text'].apply(lambda x:LexicalRichness(x).hdd(draws=1))
df_t_p_t.head(2)

# %% [cell 73: markdown]
# # Finding clauses

# %% [cell 74: code]
# -------------------- REPLACEMENT: high-precision, order-preserving clause splitter --------------------
BOUNDARY_PUNCT = {",", ";", "—", "–", ":"}
COORD_WORDS    = {"and","but","or","so"}
SUBORD_WORDS   = {"because","although","though","while","when","if","that","who","which"}
REL_WORDS      = {"that","who","which"}

MIN_TOK = 4
VERB_LOOKAHEAD = 6

def _is_finite_verb(tok):
    if tok.pos_ in ("VERB", "AUX"):
        if tok.tag_ in ("VBD","VBP","VBZ","MD"):
            return True
        m = tok.morph
        if "VerbForm=Fin" in m or "Tense=Past" in m or "Tense=Pres" in m:
            return True
    return False

def _has_finite(span):
    return any(_is_finite_verb(t) for t in span)

def _verb_within(toks, start, k):
    end = min(len(toks), start + k)
    return any(_is_finite_verb(t) for t in toks[start:end])

def _is_boundary_token(t):
    if t.text in BOUNDARY_PUNCT:
        return True
    if t.pos_ == "CCONJ" or t.lemma_.lower() in COORD_WORDS:
        return True
    if t.lemma_.lower() in SUBORD_WORDS:
        return True
    return False

def find_clauses_in_order(text, nlp=None, keep_cc=True):
    """
    Order-preserving clause splitter with correct handling of end-of-sentence markers
    (e.g., 'because' at EOS stays with the left clause) and contractions.
    """
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", exclude=["ner","textcat","lemmatizer"])

    doc = nlp(text)
    clauses = []

    for sent in doc.sents:
        toks = list(sent)
        start = 0
        has_finite = False
        spans = []

        i = 0
        while i < len(toks):
            t = toks[i]
            if _is_finite_verb(t):
                has_finite = True

            commit = False
            reason = None

            # A) explicit boundary tokens
            if _is_boundary_token(t):
                lem = t.lemma_.lower()
                if lem in COORD_WORDS:
                    if has_finite and _verb_within(toks, i+1, VERB_LOOKAHEAD) and (i - start) >= (MIN_TOK - 1):
                        commit = True; reason = "boundary"
                elif lem in SUBORD_WORDS:
                    if has_finite and (i - start) >= (MIN_TOK - 1):
                        commit = True; reason = "boundary"
                elif t.text in BOUNDARY_PUNCT:
                    if has_finite and (i - start) >= (MIN_TOK - 2):
                        commit = True; reason = "boundary"

            # B) new subject begins a finite clause: split BEFORE the subject (fixes "she's" cases)
            if (not commit) and t.dep_ in {"nsubj","nsubjpass","expl"}:
                head = t.head
                if head.i > t.i and _is_finite_verb(head) and has_finite and (i - start) >= (MIN_TOK - 1):
                    commit = True; reason = "new_subject"

            # C) end of sentence always commits
            if i == len(toks) - 1:
                commit = True
                if reason is None:
                    reason = "eos"

            if commit:
                marker_lemma = toks[i].lemma_.lower()

                # Decide span end:
                if reason == "new_subject":
                    end = i                                # cut BEFORE subject
                elif reason == "eos":
                    end = i + 1                           # include final token (e.g., 'because') with LEFT
                elif reason == "boundary":
                    # Keep subordinators/relatives with RIGHT clause when there is one
                    if marker_lemma in (REL_WORDS | SUBORD_WORDS):
                        end = i                           # cut BEFORE marker
                    else:
                        end = i + 1                       # include CC/punct with LEFT
                else:
                    end = i + 1

                if end > start:
                    span = sent[start:end]
                    stext = re.sub(r"\s+", " ", span.text.strip())
                    if not keep_cc:
                        stext = re.sub(r"^(?:and|but|or|so)\s+", "", sext, flags=re.I)  # noqa: F821 (typo fix below)
                    if stext:
                        spans.append([start, end, stext])

                # Decide next start:
                if reason == "new_subject":
                    start = i                              # next clause starts at the subject
                elif reason == "boundary" and marker_lemma in (REL_WORDS | SUBORD_WORDS):
                    start = i                              # right clause begins with the marker (that/who/because)
                else:
                    start = i + 1

                has_finite = False

            i += 1

        # Merge tiny/non-finite spans forward or backward
        merged = []
        for j, (a, b, stext) in enumerate(spans):
            span = sent[a:b]
            too_short = (len(span) < MIN_TOK)
            no_finite = not _has_finite(span)
            if (too_short or no_finite) and (j < len(spans) - 1):
                na, nb, _ = spans[j+1]
                spans[j+1] = [a, nb, re.sub(r"\s+", " ", sent[a:nb].text.strip())]
            elif (too_short or no_finite) and merged:
                pa, pb, _ = merged[-1]
                merged[-1] = [pa, b, re.sub(r"\s+", " ", sent[pa:b].text.strip())]
            else:
                merged.append([a, b, stext])

        # Safety: ensure full coverage (no dropped tail)
        if merged:
            last_a, last_b, _ = merged[-1]
            if last_b < len(toks):
                merged[-1] = [last_a, len(toks), re.sub(r"\s+", " ", sent[last_a:len(toks)].text.strip())]
        elif len(toks) > 0:
            merged.append([0, len(toks), sent[0:len(toks)].text.strip()])

        # Emit
        for a, b, stext in merged:
            s = stext.strip()
            if s:
                clauses.append(s)

    return clauses


def correct_start_and_end_of_sentnece(text):
    if (text[-2:] == ' s') or (text[-2:] == ' d') or (text[-2:] == ' m'):
        text = text[:-2] + text[-1]
    elif (text[-3:] == ' nt') or (text[-3:] == ' re') or (text[-3:] == ' ve') or (text[-3:] == ' ll'):
        text = text[:-3] + text[-2:]

    if text!='' and text!="" and text!=None:
        if text[0]=="'":
            text = text[1:]
        if text[-1]=="'":
            text = text[:-1]
    return text.strip()

def delete_punctuation(text, dictionary={",":"", ".":"", "?":"", "!":""}):
    for key, value in dictionary.items():
        # print(text)
        text = text.replace(key, value)
        corr_text = correct_start_and_end_of_sentnece(text.strip())
    return corr_text

def Finding_clauses_list(text):
    en = spacy.load('en_core_web_sm')
    doc = en(text)
    #deplacy.render(doc)

    # Use improved, contiguous clause splitter
    raw_clauses = find_clauses_in_order(text, nlp=en, keep_cc=True)
    # Keep original (index, clause) shape
    chunks = [(i, c) for i, c in enumerate(raw_clauses)]

    chunks = sorted(chunks, key=lambda x: x[0])
    dictionary_clause = {"gon what na":"gonna what", "Got ta":"Gotta", "Gon na":"Gonna", "got ta":"gotta", "gon na":"gonna",
                         " nt ":"nt ", " re ": "re ", " m ":"m ", " ve ":"ve ", " d ":"d ", " ll ":"ll ",
                         " 've":"'ve", " n't":"n't", " 'm":"'m", " 's":"'s", " s ":"s ", " 're":"'re", " 'll":"'ll", " 'd":"'d",
                         " ’ve":"’ve", " n’t":"n’t", " ’m":"’m", " ’s":"’s", " ’re":"’re", " ’ll":"’ll", " ’d":"’d",
                         ",":"", ".":"", "?":"", "!":"", "  ":" ", 'am I gonna what':'what am I gonna', 'gon that na':'that gonna'}
    clauses_list = [delete_punctuation(clause, dictionary_clause) for ii, clause in chunks]
    # for clause in clauses_list: #ali
    #     print(clause)           #ali
    return clauses_list

# %% [cell 75: code]
def similarity_score_between_text(df, df_t_p_t, file_name, postfix_col='without stop word'):
    similarity_scores = []
    if df.shape[0] <= 1:
        df_t_p_t.loc[file_name, 'Mean similarity' + ' ' + postfix_col] = 0
        df_t_p_t.loc[file_name, 'Std similarity' + ' ' + postfix_col] = 0
        df_t_p_t.loc[file_name, 'Proportion zero similarity' + ' ' + postfix_col] = 0
    else:
        for i in range(df.shape[0]):
            for j in range(i+1, df.shape[0]):
                text_1, text_2 = df.loc[i, 'lemma'], df.loc[j, 'lemma']
                # Convert the texts into TF-IDF vectors
                vectorizer = TfidfVectorizer()
                try:
                    vectors = vectorizer.fit_transform([text_1, text_2])
                except ValueError:
                    continue
                # Calculate the cosine similarity between the vectors
                similarity = cosine_similarity(vectors)
                similarity_scores.append(similarity[0, 1])
        df_t_p_t.loc[file_name, 'Mean similarity' + ' ' + postfix_col] = np.mean(similarity_scores)
        df_t_p_t.loc[file_name, 'Std similarity' + ' ' + postfix_col] = np.std(similarity_scores)
        similarity_scores = np.array(similarity_scores)
        df_t_p_t.loc[file_name, 'Proportion zero similarity' + ' ' + postfix_col] = len(np.where(similarity_scores==0.0)[0])/len(similarity_scores)
    return df_t_p_t

def matching_clauses_to_dataframe(df, clauses_list):
    # print("All clause:", clauses_list) #ali
    # print("clause_list:\n")
    text_remove_punc = ' '.join(df['content_remove_punc'].tolist())
    # print("Main text:", text_remove_punc)
    remain_clause_words_list = []
    df['clauses'] = np.nan
    for i, clause in enumerate(clauses_list):
        # print(i, clause)
        clause_words = clause.split()
        if clause in text_remove_punc and clause:
            # print(f"Candidate clause {i+1}:", clause) #ali
            # split_list = text_remove_punc.replace("'", "").split(clause) #ali
            split_list = text_remove_punc.split(clause)
            # for repetitive clauses or there are this clause in other clauses.
            if (len(split_list) > 2):
                remain_clause_words_list.append((i, clause_words))
                # print('clause_' + str(i+1))
                # print('******************************')
                continue
            text_remove_punc = ''
            row_start = len(split_list[0].split())
            row_end = len(split_list[0].split()) + len(clause_words) - 1
            text_remove_punc += split_list[0] + ' '.join(['Claused_filled']*len(clause_words)) + split_list[1]
            df.loc[row_start:row_end, 'clauses'] = 'clause_' + str(i+1)

        else:
            remain_clause_words_list.append((i, clause_words))
            # print(remain_clause_words_list) #ali
            # print()
            # print('clause_' + str(i+1))
        # print('******************************')
    # print(remain_clause_words_list)
    # display(df)
    for ii, clause_words in remain_clause_words_list:
        # print(clause_words)
        df_isnan = df[df['clauses'].isna()]
        df_contain = df_isnan[df_isnan['content_remove_punc'].isin(clause_words)]
        # display(df_contain) #ali

        # display(df_contain)
        # print('Clauses__________________:', ' '.join(clause_words))
        # print('All words without clauses:', ' '.join(df_contain['content_remove_punc'].tolist()))
        # display(df_contain)
        # When reversing on word is happend
        if len(clause_words) == df_contain.shape[0]:
            df.loc[df_contain.index.tolist(), 'clauses'] = 'clause_' + str(ii+1)
            continue
        index_list_order = []
        for i, word in enumerate(clause_words):

            # print("word in clause:", word)
            # print('**********************************')

            index_list = df_contain[df_contain["content_remove_punc"]==clause_words[i]].index.tolist()
            # print(index_list)

            if i==0:
                # Minus one just to first_index_list should be greater than before_first_index_list
                index_list_prev = deepcopy(np.array(index_list)-1)
            if i<len(clause_words)-1:
                index_list_next = df_contain[df_contain["content_remove_punc"]==clause_words[i+1]].index.tolist()
            new_index_list = []
            for idx in index_list:
                if len(clause_words)==1:
                    new_index_list.append(idx)
                    break
                if i==0:
                    if (idx < np.array(index_list_next)).any():
                        new_index_list.append(idx)
                elif i==len(clause_words)-1:
                    if (idx > np.array(index_list_prev)).any():
                        new_index_list.append(idx)
                else:
                    if (idx < np.array(index_list_next)).any() and (idx > np.array(index_list_prev)).any():
                        new_index_list.append(idx)
            if len(new_index_list)!=0:
                index_list_prev = deepcopy(new_index_list)
                index_list_order.append(new_index_list)
            # added lastly
            else:
                index_list_order.append(index_list)
            # print("index_list_order:", index_list_order) #ali
            # print("************") #ali

        while True:
            len_len_index_list_order = len(np.where(np.array([len(idxs) for idxs in index_list_order])>1)[0])
            print(len_len_index_list_order)
            # print(index_list_order)
            if len_len_index_list_order==0:
                break
            for i in range(len(index_list_order)):
                if len(index_list_order[i])> 1:
                    if i==0:
                        if len(index_list_order[i+1]) > 1:
                            continue
                        distance = np.array(index_list_order[i]) - index_list_order[i+1][0]
                        index_list_order[i] = [index_list_order[i][np.argmin(distance)]]
                    elif i==len(index_list_order)-1:
                        distance = np.array(index_list_order[i]) - index_list_order[i-1][0]
                        index_list_order[i] = [index_list_order[i][np.argmin(distance)]]
                    else:
                        if len(index_list_order[i-1])==1:
                            distance = np.array(index_list_order[i]) - index_list_order[i-1][0]
                            index_list_order[i] = [index_list_order[i][np.argmin(distance)]]
                        elif len(index_list_order[i+1])==1:
                            distance = np.array(index_list_order[i]) - index_list_order[i+1][0]
                            index_list_order[i] = [index_list_order[i][np.argmin(distance)]]

        flat_index_list = [i for sublist in index_list_order for i in sublist]  # Flatten the list
        # print(flat_index_list)#ali
        df.loc[flat_index_list, 'clauses'] = 'clause_' + str(ii + 1)
        # display(df)
    # for filling one row that have nan value in clause column
    df_isnan_final = df[df['clauses'].isna()]
    if df_isnan_final.shape[0]==1:
        index_nan = df_isnan_final.index.tolist()[0]
        clause_num = df.loc[index_nan - 1, 'clauses']
        clause_num_split = clause_num.split('_')
        df.loc[index_nan, 'clauses'] = clause_num_split[0] + '_' + str(int(clause_num_split[1])-1)
    elif df_isnan_final.shape[0]>1:
        index_nan = df_isnan_final.index.tolist()
        index_nan_first = index_nan[0]
        index_nan_last = index_nan[-1]
        clause_num_before_first = df.loc[index_nan_first - 1, 'clauses']
        clause_num_after_last = df.loc[index_nan_last + 1, 'clauses']
        if clause_num_before_first == clause_num_after_last:
            df.loc[index_nan, 'clauses'] = clause_num_before_first
    return df.copy()

def add_space_after_punctuation(text):
    pattern = r'([,.?!])(?=\S)'  # Match a comma, period, or question mark followed by a non-space character
    replaced_text = re.sub(pattern, r'\1 ', text)  # Replace the punctuation with punctuation + space
    return replaced_text

def compute_clause_silence_pos(df_t_p_t):
    for i, idx in enumerate(df_t_p_t.index):
        df = pd.DataFrame(columns=['content'])
        # Read files
        dict_modify = {"*":"",  "#":"", '2-2':'Two', '->':'', "–":"", '-':'', "’":"'", 'wed':"we'd","mightve":"might've", "mayve":"may've", "nt s ":"nt ",
                       "@":"", "$":"", "%":"", "^":"", "&":"", "/":"", "|":"", " s ": "s ", " re ":"re ", " m ":"m ", " d ": "d ", " ve ":"ve ", " ll ":"ll ",
                       ' junʌŋˈjʌnzﬨt':'', '.,':'', ',.':'', '.?':'', '?.':'', "o'clock":"oclock", "„":"", ',':'',
                       '"':'', "\n":"", ' ...':'', '....':'', '...':'', '..':'', '…':'',  '—':'', "“":"", "”":"", "‘":"", "’":"", "  ":" ", "   ":" ", "''":"", '""':'',
                       "(":"", ")":"", "[":"", "]":"", ":":"", ";":"", "' ":" ", "'  ":" ", " '":" ", "in'":"in", ",'":",", ".'":".", "?'":"?", "!'":"!",
                       " . ":" ", " ? ":" ", " ! ":" ", " , ": " "}

        df_t_p_t.loc[idx, 'text'] = " ".join(w.strip() for w in df_t_p_t.loc[idx, 'text'].split())

        for key, value in dict_modify.items():
            text_value = df_t_p_t.loc[idx, 'text']
            if isinstance(text_value, pd.Series):
                text_value = text_value.iloc[0]  # Extract single value
            df_t_p_t.loc[idx, 'text'] = text_value.strip(string.punctuation).replace(key, value).strip()

        df_t_p_t.loc[idx, 'text'] = " ".join(w.strip() for w in df_t_p_t.loc[idx, 'text'].split())
        # print(df_t_p_t.loc[idx, 'text']) #ali


        text_value = df_t_p_t.loc[idx, 'text']
        if isinstance(text_value, pd.Series):
            text_value = text_value.iloc[0]  # Extract single value

        # Replace the punctuation with punctuation(.,!?) + space
        # print(text_value) #ali
        text_preprocessed = add_space_after_punctuation(text_value)
        # Correct the end of sentence
        text_preprocessed = correct_start_and_end_of_sentnece(text_preprocessed)
        df['content'] = text_preprocessed.split(' ')
        # display(df)
        # Get name of files
        file_name = idx
        print(f"{i} - {file_name}:\n")

        # # Silence time
        # df['silence_time'] = (df['start_time'] - df['end_time'].shift(1, fill_value=0)).abs()
        # Finding and matching clauses
        df['content'] = df['content'].astype(str)
        text = ' '.join(df['content'].tolist())

        clauses_list = Finding_clauses_list(text)
        # print("clauses_list:")  #ali
        # print(clauses_list)     #ali
        # print("Initial clause list:", clauses_list) #ali
        df['content_remove_punc'] = df['content'].apply(delete_punctuation)
        # display(df) #ali
        # try:
        df = matching_clauses_to_dataframe(df, clauses_list)
        # display(df) #ali
        # Finding POS, Lemma, Stop word, Dependency
        nlp = en_core_web_sm.load()
        # "nlp" Objectis used to create documents with linguistic annotations.
        text_remove_punc = ' '.join(df['content_remove_punc'].tolist())
        row = 0
        # iii = 0
        for token in nlp(text_remove_punc):
            # print("{} : {} : {}".format(iii, token.text, type(token.text)))
            # iii +=1
            # yasaman modify
            # if (row!=0 and len(df.loc[row-1, 'pos'].split())==1) and (df.loc[row-1, 'content_remove_punc']=='gonna' or df.loc[row-1, 'content_remove_punc']=='gotta') and ("'s" in token.text):
            if (row!=0 and len(df.loc[row-1, 'pos'].split())==1) and (str(df.loc[row-1, 'content_remove_punc']).lower()=='gonna' or str(df.loc[row-1, 'content_remove_punc']).lower()=='gotta' or str(df.loc[row-1, 'content_remove_punc']).lower()=='cannot')\
                or ("'" in token.text) or (token.text=='s') or (token.text=='nt') or (token.text=='re') or (token.text=='m') or (token.text=='d') or (token.text=='ll') or (token.text=='ve'):
                # print(df[row-1])
                df.loc[row-1, 'pos'] = df.loc[row-1, 'pos'] + ' ' + token.pos_
                df.loc[row-1, 'lemma'] = df.loc[row-1, 'lemma'] + ' ' + token.lemma_
                df.loc[row-1, 'dependency'] = df.loc[row-1, 'dependency'] + ' ' + token.dep_
                continue
            else:
                df.loc[row, ['pos', 'stop_word', 'lemma', 'dependency']] = \
                                      {'pos':token.pos_, 'stop_word':token.is_stop, 'lemma':token.lemma_, 'dependency':token.dep_}
            row += 1
        # except:
        #     print('This index has problem:', idx)
        #     continue
        # Save dataframe
        # df.to_excel(path_file, index=False)

        # # Compute silent clause for each file
        # within_clauses_silence_sum = df.groupby('clauses', as_index=False, sort=False)['silence_time'].mean()['silence_time'].sum()
        # df_t_p_t.loc[file_name, 'Sum of mean silence time per word within clauses'] = within_clauses_silence_sum
        # initial_clauses_silence_sum = df.groupby('clauses', as_index=False, sort=False)['silence_time'].first()['silence_time'].sum()
        # df_t_p_t.loc[file_name, 'Sum of mean silence time initial clauses'] = initial_clauses_silence_sum
        # # 'NOUN'
        # within_clauses_silence_of_nouns = df[df['pos']=='NOUN'].groupby('clauses', as_index=False, sort=False)['silence_time'].mean()['silence_time'].sum()
        # df_t_p_t.loc[file_name, 'Sum of mean silence time for NOUN within clauses'] = within_clauses_silence_of_nouns
        # # 'VERB'
        # within_clauses_silence_of_verbs = df[df['pos']=='VERB'].groupby('clauses', as_index=False, sort=False)['silence_time'].mean()['silence_time'].sum()
        # df_t_p_t.loc[file_name, 'Sum of mean silence time for VERB within clauses'] = within_clauses_silence_of_verbs
        # # 'ADJ', 'ADV'
        # within_clauses_silence_of_adjs_advs = df[df['pos'].isin(['ADJ', 'ADV'])].groupby('clauses', as_index=False, sort=False)['silence_time'].mean()['silence_time'].sum()
        # df_t_p_t.loc[file_name, 'Sum of mean silence time for ADJ ADV within clauses'] = within_clauses_silence_of_adjs_advs
        # Compute similarity between clauses
        ## Without stop words
        df_not_stop = df[df['stop_word']==False]
        df_not_stop_clause = df_not_stop[['clauses', 'lemma']].groupby('clauses', as_index=False, sort=False).agg(lambda x:x.str.cat(sep=' '))
        df_t_p_t = similarity_score_between_text(df_not_stop_clause, df_t_p_t, file_name, postfix_col='without stop word')
        ## With stop words
        df_clause = df[['clauses', 'lemma']].groupby('clauses', as_index=False, sort=False).agg(lambda x:x.str.cat(sep=' '))
        df_t_p_t = similarity_score_between_text(df_clause, df_t_p_t, file_name, postfix_col='with stop word')
        display(df[df['clauses'].isna()])
        # display(df)
        # print('************************\n')
    return df, df_t_p_t.reset_index()

# %% [cell 76: code]
df_lasr_par, df_t_p_t_final = compute_clause_silence_pos(df_t_p_t.iloc[:, :])

# %% [cell 77: markdown]
# # Save Result

# %% [cell 78: code]
df_t_p_t_final.head(3)

# %% [cell 79: code]
if 'label' in df_t_p_t_final.columns:
    label_name = 'label'
elif 'Label' in df_t_p_t_final.columns:
    label_name = 'Label'
df_lingustic = pd.concat([df_t_p_t_final[[label_name]], df_t_p_t_final.loc[:, "Content Density":"Proportion zero similarity with stop word"]], axis=1)
df_lingustic.to_excel(DATA_PATH + './Handcrafted_fr_train_df_story_recall_cinderella.xlsx', index=False)

# %% [cell 80: code]

