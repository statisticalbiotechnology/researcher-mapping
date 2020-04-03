import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import collections
import gensim
from sklearn.manifold import TSNE
import pandas as pd
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, value


df = pd.read_csv("assets/dataframes/all_authors_df_2004_small")

text_doc, doc_id, dep, auth, name = df.Abstracts.values, df.Doc_id, df.department, df.KTH_id, df.KTH_name

def read_corpus(abstracts, doc, auth):
    for d, w, a in zip(abstracts, doc, auth):
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(d), [str(w)] + a.split(":"))
                         

train_corpus = list(read_corpus(text_doc, doc_id, auth))

import multiprocessing

cores = multiprocessing.cpu_count()

model = gensim.models.doc2vec.Doc2Vec(vector_size=500, min_count=1,dm=0,sample=1e-3, negative=15,hs=0,dbow_words=1,max_vocab_size=None,workers=cores,window=10)
model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=2000, report_delay=1)

model.save("assets/doc2vecModels/KTH2004_i2000_w10_d500_plainTrain_small/KTH2005_i2000_w10_d500_plainTrain")
