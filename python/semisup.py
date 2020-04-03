from utils import *
import gensim
from sklearn.mixture import BayesianGaussianMixture
import json

from utils import *

import gensim
from sklearn.mixture import BayesianGaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from semiSupervisedDnn import selfTrainer

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

df = pd.read_csv("assets/finalproduct/finalproductDf")
df.drop(["Unnamed: 0"],axis=1, inplace=True)
id_to_auth = pickle_o.load("assets/dictionaries/id_to_all_auths_2004")
auth_to_id = pickle_o.load("assets/dictionaries/auths_to_all_id_2004")

Name = list(df.Author.values)
kth_id = [auth_to_id[a] for a in Name]
df_only_auth = pd.DataFrame(data={"Name":Name, "ID":kth_id})
df_only_auth.to_csv("assets/finalproduct/onlyAuthors.csv")

df_abs = pd.read_csv("assets/dataframes/all_authors_df_2004")
df_abs.drop(["Unnamed: 0"],axis=1, inplace=True)

all_auth = pd.read_csv("assets/dataframes/KTH_UPPSALA_SODERTORN_SU")
all_auth.drop(["Unnamed: 0"],axis=1, inplace=True)

model = gensim.models.Word2Vec.load("assets/doc2vecModel/more_school_1000")

doc_tag = list(model.docvecs.doctags.keys())

kth_id = list()
for d in all_auth.Doc_id:
    try:
        kth_id.append(int(d))
    except:
        pass

ls_d = list()
for d in df.Doc_id.values:
    ls_d += d.split(":")
    

train_label = list()
for d in kth_id:
    if str(d) in ls_d:
        train_label.append(0)
    else:
        train_label.append(1)

kth_vec = model[np.array(kth_id).astype(str)]
kth_label = np.asarray(train_label)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(kth_vec, kth_label, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)



b = GaussianNB()
b.fit(X_train, y_train)

test_pred = b.predict(X_test)
print(accuracy_score(test_pred, y_test))


article_tag = [t for t in doc_tag if not t[0].islower()]

unlabeled_tags = list(set(article_tag) - set(np.array(kth_id).astype(str)))

unlabeled_vec = model[unlabeled_tags]


sel_train = selfTrainer(topk=1, save_error=True, epoch=150, batch_size=256,
                        verbose=True, keras=False)
sel_train.fit(X_train, y_train, X_test, y_test, unlabeled_vec,X_val, y_val)

