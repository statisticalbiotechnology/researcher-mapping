import pandas as pd
import gensim
import multiprocessing
import numpy as np
from utils import pickle_obj, semantic_search_author, semantic_search_word, get_related_authors, get_related_words, translate_dict
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show, output_notebook, output_file, save
from bokeh.models import HoverTool, ColumnDataSource, value
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds

df_kth = pd.read_csv("assets/dataframes/all_authors_df_2004")
df_su = pd.read_csv("assets/dataframes/suDf")
df_uppsala = pd.read_csv("assets/dataframes/uppsalaDf")
df_sodertorn = pd.read_csv("assets/dataframes/sodertornDf")

df_kth = df_kth.rename(columns={"KTH_id": "Auth_id", "KTH_name": "Auth_name"})

def get_nlp_data(df):
    return df.Abstracts.values, df.Doc_id.values, df.Auth_id.values, df.Auth_name.values

text_doc_kth, doc_id_kth, auth_kth, name_kth = get_nlp_data(df_kth)
text_doc_su, doc_id_su, auth_su, name_su = get_nlp_data(df_su)
text_doc_uppsala, doc_id_uppsala, auth_uppsala, name_uppsala = get_nlp_data(df_uppsala)
text_doc_sodertorn, doc_id_sodertorn, auth_sodertorn, name_sodertorn = get_nlp_data(df_sodertorn)

TEXT = np.concatenate([text_doc_kth, text_doc_su, text_doc_uppsala, text_doc_sodertorn])
DOCID = np.concatenate([doc_id_kth, doc_id_su, doc_id_uppsala, doc_id_sodertorn]).astype(str)
AUTHID = np.concatenate([auth_kth, auth_su, auth_uppsala, auth_sodertorn ])
NAME = np.concatenate([name_kth, name_su, name_uppsala, name_sodertorn ])

df = pd.DataFrame(data=list(zip(TEXT, AUTHID, DOCID, NAME)), columns=["Abstracts", "Auth_id", "Doc_id", "Auth_name"])

def read_corpus(abstracts, doc):
    for d, w in zip(abstracts, doc):
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(d), [str(w)])

train_corpus = list(read_corpus(TEXT, DOCID))

from gensim.test.utils import get_tmpfile
from gensim.models.callbacks import CallbackAny2Vec


class EpochSaver(CallbackAny2Vec):
    '''Callback to save model after each epoch.'''
    
    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_end(self, model):
        output_path = get_tmpfile('{}_epoch{}.model'.format(self.path_prefix, self.epoch))
        model.save(output_path)
        self.epoch += 1
        
class EpochLogger(CallbackAny2Vec):
    '''Callback to log information about training'''
    
    def __init__(self):
        self.epoch = 0
        
    def on_epoch_begin(self, model):
        print("Epoch #{} start".format(self.epoch))
    
    def on_epoch_end(self, model):
        print("Epoch #{} end".format(self.epoch))
        self.epoch += 1


epoch_logger = EpochLogger()

cores = multiprocessing.cpu_count()

model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=1, dm=0,
                                      sample=1e-3, negative=15,hs=0,dbow_words=1,
                                      max_vocab_size=None,workers=cores,window=10,
                                          callbacks=[epoch_logger])

model.build_vocab(train_corpus)


import time

start = time.time()
model.train(train_corpus, total_examples=model.corpus_count, epochs=1000,report_delay=1)
end = time.time()
print(end - start)


from gensim.test.utils import get_tmpfile

fname = get_tmpfile("doc2vec_more_school_1000_onlyDocId")

model.save(fname)
