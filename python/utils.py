import pickle
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import itertools
#from semiSupervised import *
import string
import time

def sample_abstract(cluster_nr, df,df_abs, max_nr_article=10, return_ix=True, keep_count=None, treshold=0):
    nr_article, article_id = get_nr_cluster(cluster_nr, df)
    all_ix = get_ix(nr_article, max_nr_article)
    IX = list()
    art_id = list()
    for i, (ix, ixx) in enumerate(zip(all_ix, article_id)):
        if return_ix:
            if i >= treshold:
                print("------------", i + keep_count, "   Article number:", str(int(float(ixx))))
                see_abstract(cluster_nr, ix, df, df_abs)
        
        IX.append(i + keep_count)
        if i >= treshold:
            art_id.append(str(int(float(ixx))))
    return IX, art_id

def get_article_in_cluster(df):
    invalidChars = set(string.punctuation.replace("_", ""))
    df_articles = pd.DataFrame()
    length_list = list()
    for i, col in enumerate(df.columns):
        id_list = list()
        
        for row in df[col].dropna().tolist():
            if row is np.nan:
                break
            else:
                if not str(row).islower():
                    try:
                        id_list.append(row)
                    except:
                        print(row)
                else:
                    break
        length_list.append(len(id_list))
        df_new = pd.DataFrame({int(col): id_list})
        df_articles = pd.concat([df_articles, df_new], axis=1)
    return df_articles, np.array(length_list)


def print_abstract(idx, df_abs): print(df_abs[df_abs.Doc_id == idx].Abstracts.values[0])

def RepresentsInt(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_nr_cluster(cluster, df):
    nr_article = list()
    article_id = list()
    for i, v in enumerate(df[str(cluster)].dropna()):
        v = str(v)
        if not v.islower():
            nr_article.append(i)
            article_id.append(v)
    return nr_article, article_id

def get_ix(nr_total_article, nr_article=10, seed=7):
    ix = np.arange(len(nr_total_article))
    np.random.seed(seed)
    np.random.shuffle(ix)
    return nr_total_article[:nr_article]

def see_abstract(cluster, article_nr, df, df_abs):
    ix = df[str(cluster)][article_nr]
    print(df_abs[df_abs.Doc_id == int(ix)].Abstracts.values)

class get_author_lits(object):
    def get_words(self, arr):
        words = list()
        for v in arr:
            if not v[0].isupper():
                words.append(v)
        return words
    def get_author(self, arr):
        author = list()
        for v in arr:
            if v in list(id_to_auth.keys()):
                author.append(v)
        return author
    def get_author_lits(self, cluster_list, df):
        author_list = list()
        for cluster in cluster_list:
            author_list.append(self.get_author(df[int(cluster)].dropna().values))
        return author_list


def convert_lists_2_df(lists):
    df = pd.DataFrame()
    for i, x in enumerate(lists):
        df_new = pd.DataFrame({'cluster' + str(i):x})
        df = pd.concat([df, df_new], ignore_index=True, axis=1)
    return df

def show_abstrcts(name, df):
    print(df[df.KTH_name.str.contains(name)].values)

def get_related_authors(name, model, auth_to_id, id_to_auth, topn=50):    
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """  
    author_Series = pd.Series(list(auth_to_id.keys()))
    if not any(author_Series == name):
        check_name = author_Series.str.contains(str(name))
        if any(check_name):
            print("Did you mean any ofese name/s:")
            for a in author_Series[check_name].unique(): 
                print(a)
        else:
            print("Couldn't find any good mathc")
    else:
        tag = auth_to_id[name]
        
        for r in model.docvecs.most_similar(tag,topn=topn):
            if r[0][0] == "u":
                print(id_to_auth[r[0]],r[1])

def get_related_words(name, model, auth_to_id, id_to_auth, topn=50):
    """
    look up the topn most similar terms to token
    and print them as a formatted list
    """  
    author_Series = pd.Series(list(auth_to_id.keys()))
    if not any(author_Series == name):
        check_name = author_Series.str.contains(str(name))
        if any(check_name):
            print("Did you mean any ofese name/s:")
            for a in author_Series[check_name].unique(): 
                print(a)
        else:
            print("Couldn't find any good mathc")
    else:
        tag = auth_to_id[name]
        tag_v = model[str(tag)]
        for r in model.most_similar([tag_v], topn=topn):
            print(r)

def semantic_search_author(sentence, model, df, topn=30):
    word_2_vec = 0
    for word in sentence:
        word_2_vec += model[str(word)]
    for a in model.docvecs.most_similar( [ word_2_vec ], topn=topn):
        if a[0][0] !="u":
            print(str(a[0]),"||",get_article_authors_name(int(a[0]),df)," || ",np.around(a[1],2))

def semantic_search_word(sentence, model, df, topn=30):
    word_2_vec = 0
    for word in sentence:
        word_2_vec += model[str(word)]
    for a in model.most_similar( [ word_2_vec ], topn=topn):
        if a[0] not in sentence:
            print(a)

def get_article_authors_name(doc_id, df):
    name = ""
    if len(df[df.Doc_id == doc_id].KTH_name) > 0:
        name_list = df[df.Doc_id == doc_id].KTH_name.values[0].split(":")
        for i, n in enumerate(name_list):
            if i == 0:
                name += str(n)
            elif i == len(name_list) - 1 and i > 1:
                name += " and " + str(n)
            elif i > 1:
                name += " , " + str(n)
    else:
        name = "NaN"
        
    return name

class pickle_obj:
    def save(self, obj, name ):
        with open(name + '.pkl', 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    def load(self, name ):
        with open(name + '.pkl', 'rb') as f:
            return pickle.load(f)

def save_classifer(model, path):
    # save the classifier
    with open(path + '.pkl', 'wb') as fid:
        pickle.dump(model, fid)

def load_classifier(path):        
    # load it again
    with open(path + '.pkl', 'rb') as fid:
        BGMM_loaded = pickle.load(fid)
    return BGMM_loaded

translate_dict = dict();
translate_dict["ö"] = "o"
translate_dict["ä"] = "a"
translate_dict["å"] = "a"
translate_dict["Ö"] = "O"
translate_dict["Ä"] = "A"
translate_dict["Å"] = "A"
translate_dict["é"] = "e"
translate_dict["ü"] = "u"
translate_dict["á"] = "a"
translate_dict["Á"] = "A"
translate_dict["\xad"] = ""
translate_dict["æ"] = "a"
translate_dict["¡"] = "i"
translate_dict["«"] = "a"
translate_dict["ó"] = "O"
translate_dict["œ"] = "o"
translate_dict["\xa0"] = ""
translate_dict["ç"] = "c"
translate_dict["ñ"] = "n"
translate_dict["ú"] = "u"
translate_dict[","] = ""
translate_dict["è"] = "e"
translate_dict["‰"] = ""
translate_dict["‰"] = ""
translate_dict["£"] = "E"
translate_dict["“"] = ""
translate_dict["ˆ"] = ""
translate_dict["\x8d"] = ""
translate_dict["™"] = ""
translate_dict["²"] = ""
translate_dict["Ì"] = "I"
translate_dict["\x81"] = ""
translate_dict["Œ"] = "E"
translate_dict["´"] = ""
translate_dict["¸"] = ""
translate_dict["…"] = ""
translate_dict["ø"] = "o"


def make_name_noAscii(ascii_name):
    name = str()
    for l in ascii_name:
        if ord(l) < 128:
            name +=l
        else:
            name += translate_dict[l]
    return name

def fk(values, keys, C):
    # For the first 10 clusters
    for cluster in C:
        return list(keys[values==cluster])

def get_cluster_members(num_clusters, word_centroid_map):
    chunks = [[i] for i in range(0, num_clusters)]
    pool = Pool(processes=cpu_count() - 1)
    values = np.array(list(word_centroid_map.values()))
    keys = np.array(list(word_centroid_map.keys()))
    func = partial(fk, values, keys)
    result = pool.map_async(func, chunks)
    return result.get()


def get_cluster_containing_substring(S,dataFrame):
    mask = np.column_stack([dataFrame[col].astype(str).str.contains(str(S), na=False) for col in dataFrame])
    cluster = list()
    for i, col in enumerate(dataFrame):
        if dataFrame[col].astype(str).str.contains(str(S), na=False).any():
            cluster.append(col)
    return cluster

pickle_o = pickle_obj(); 
id_to_auth = pickle_o.load("assets/dictionaries/id_to_all_auths_2004")
auth_to_id = pickle_o.load("assets/dictionaries/auths_to_all_id_2004")
