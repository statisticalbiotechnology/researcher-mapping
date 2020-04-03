import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score

""" from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.metrics import f1_score
from keras.layers.normalization import BatchNormalization """

""" def get_dnn():
    # import BatchNormalization


    # instantiate model
    model = Sequential()

    # we can think of this chunk as the input layer
    model.add(Dense(500, input_dim=500, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    # we can think of this chunk as the hidden layer    
    model.add(Dense(200, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    # we can think of this chunk as the output layer
    model.add(Dense(20, kernel_initializer='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.5))

    # we can think of this chunk as the output layer
    model.add(Dense(1, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))


    # setting up the optimization of our weights 
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd,
                    metrics=['accuracy'])
    return model """

def get_dnn():
    dnn = Sequential()
    dnn.add(Dense(500, input_dim=500, activation='sigmoid'))
    dnn.add(Dropout(0.2))
    dnn.add(Dense(200, activation='sigmoid'))
    dnn.add(Dropout(0.2))
    dnn.add(Dense(50, activation='sigmoid'))
    dnn.add(Dropout(0.2))
    dnn.add(Dense(1, activation='sigmoid'))

    dnn.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
 
    
    return dnn

class selfTrainer:
    """Semi-supervised learning."""
    def __init__(self, topk=100, save_error=False
                 ,verbose=False, specifiedModel=None,
                 epoch=20, batch_size=128, keras=True):
        """
        Args:
            topk (int): Add topk certain predictions to dataset.
            save_error (boolean): Save error
            verbose (boolean): Print progress
            epoch (int): Epochs (if using keras model)
            batch_size (int): Batch size if (if using keras model)
            keras (boolean): Using keras model
        """
        self.topk = topk
        self.save_error = save_error
        self.verbose = verbose
        self.epoch = epoch
        self.batch_size = batch_size
        self.keras = keras
        if keras:
            self.specifiedModel = get_dnn()
        else:
            self.specifiedModel = GaussianNB()
            

    def _init_train(self, X_train, y_train):
        """Init training data.
        Args:
            X_train (arr): Training data
            y_train (arr): Training labels
        Returns:
            void
        """
        self.X_init = X_train.copy()
        self.y_init = y_train.copy().reshape(-1,1)
    def _init_unlabeled(self, X_unlabeled):
        """Init unlabeled data.
        Args:
            X_unlabeled (arr): Unlabeled data
        Returns:
            void
        """
        self.X_un_init = X_unlabeled.copy()
    
    def _init_model(self, X, y):
        """Init model
        Args:
            X (arr): Trainig data
            y (arr): Labeled data
        Returns:
            void
        """
        self.model = self.specifiedModel
        if self.keras:
            self.model.fit(X, y.ravel(), epochs=self.epoch,
                        batch_size=self.batch_size, verbose=0)
        else:
            self.model.fit(X, y.ravel())
    
    def _fit_model(self, X, y):
        """Fit model
        Args:
            X (arr): Trainig data
            y (arr): Labeled data
        Returns:
            void
        """
        if self.keras:
            self.model.fit(X, y.ravel(), epochs=self.epoch,
                           batch_size=self.batch_size, verbose=0)
        else:
            self.model.fit(X, y.ravel())

    def _get_acc_score(self, X, y):
        """Get accuracy
        Args:
            X (arr): Trainig data
            y (arr): Labeled data
        Returns:
            accuracy (float)
        """
        pred = np.around(self.model.predict(X))
        return accuracy_score(pred, y.ravel())

    def _get_f1_score(self, X, y):
        """Get f1 score
        Args:
            X (arr): Trainig data
            y (arr): Labeled data
        Returns:
            f1 (float)
        """
        pred = np.around(self.model.predict(X))
        return f1_score(pred, y.ravel())
    
    def _get_probs(self, X):
        """Get f1 score
        Args:
            X (arr): Trainig data
        Returns:
            sorted probabilities (a)rr
            labels (arr)
        """
        prob, labels = self.basemodel.predict_proba(X), self.basemodel.predict(X)
        return np.sort(prob, axis=1)[:, 1], labels
        
    def _init_save_error(self, test_error=False):
        """Init save_error array
        Args:
            test_error (bool): Save test error or validation error
        Returns:
            void
        """
        if test_error:
            self.test_error_list = list()
        else:
            self.val_error_list = list()
            
    def _add_error(self, error, test_error=False):
        """Add error
        Args:
            error (float): Error
            test_error (boolean): Save test error or validation error
        Returns:
            void
        """
        if test_error:
            self.test_error_list.append(error)
        else:
            self.val_error_list.append(error)
    
    def _error_dif(self):
        """Check difference of error between rounds
        Returns:
            void
        """
        return self.val_error_list[-1] - self.val_error_list[-2]
    def _save_model(self):
        """Save keras model
        Returns:
            void
        """
        if self.keras:
            self.model.save_weights('assets/kerasmodels/model_weights.h5')
        else:
            pass
    def _load_model(self):
        """Load keras model
        Returns:
            void
        """
        if self.keras:
            self.model.load_weights('assets/kerasmodels/model_weights.h5')
        else:
            self.model = GaussianNB()

    def _if_better_model(self):
        """Check if new model is better.
        Returns:
            boolean
        """
        return self.bestTestf1 <= self.f1Test
    def _use_test_data(self):
        """Check use test data.
        Returns:
            boolean
        """
        return self.bestTestAcc != -1
    def _init_unlabeledData_relative_ix(self, length):
        """Init the relative indices
        Args:
            length(int): length of unlabeledData
        Returns:
            void
        """
        self.relative_indices = np.arange(length)
    def _init_index_to_keep(self):
        """Init index list
        Returns:
            void
        """
        self.keep_index = list()
    def _keep_indieces(self, ix):
        """Indices to save.
        Args:
            ix(arr): array with indices
        Returns:
            void
        """
        self.keep_index += list(ix)
    def _get_probability_and_labels(self, X):
        """Get probabilty of prediction of data.
        Args:
            X(arr): data
        Returns:
            p(arr): probabilities
            label(arr): predicted labels
        """
        p = self.model.predict(X)
        labels = np.around(p)
        p[p <= .5] = 1 - p[p <= .5]
        return p, labels
    def _get_absolute_ix(self, p):
        """Get indices of most certain predictions.
        Args:
            p(arr): probabilities
        Returns:
            ix(arr): indices
        """
        sortedIx = np.argsort((p.ravel()))[::-1]
        ix = sortedIx[: self.topk]
        return ix
    def _sample_new_data(self, X):
        """Sample new data from unlabeled data.
        Args:
            X(arr): data
        Returns:
            X(arr): new datapoints
            labels(arr): new labels
            ix(arr): indices
        """
        p, labels = self._get_probability_and_labels(X)
        ix = self._get_absolute_ix(p)
        return X[ix], labels[ix], ix
    def _add_data(self, X, y, X_new_data=None, y_new_data=None, sample=False):
        """Add data to the training data.
        Args:
            X(arr): data
            y(arr): labels
            X_new_data(arr): new datapoints
            y_new_data(att): new labels
            sample(boolean): add sample or not 
        Returns:
            X(arr): Data
            y(arr): Labels
        """
        y = y.reshape(-1,1)
        if sample:
            y_new_data = y_new_data.reshape(-1,1)
            return np.concatenate((X_new_data, X)), np.concatenate((y_new_data, y))
        else:
            return np.concatenate((self.X_init, X)), np.concatenate((self.y_init, y))
    

    def _update_(self, tmp_sampleX, tmp_sampley, ix):
        """Update data and scores
        Args:
            tmp_sampleX(arr): extended training data
            tmp_sampley(arr): extended labels
            ix(arr): data
        Returns:
            tmp_sampleX(arr): extended training data
            tmp_sampley(arr): extended labels
        """
        self._save_model(), self._keep_indieces(ix)
        if self._use_test_data():
            self.bestTestAcc = self.AccTest
            self.bestTestf1 = self.f1Test
        return tmp_sampleX, tmp_sampley

    def _if_train(self, i):
        """Check if continuation of training.
        Args:
            i(int): iteration
        Returns:
            boolean
        """
        return i != -1

    def _report(self, error, iteration, new_data_n):
        """Print progress
        Args:
            error(float): Training error
            iteration(int): Training iteration
            new_data_n(int): Number of new datapoints
        Returns:
            void
        """
        print("Acc at iter {}:".format(iteration), error, "and the dif: ", self._error_dif())
        if self._use_test_data():
            print("and acc: {} , and f1: {}, and n new data: {}".format(self.bestTestAcc, self.bestTestf1, new_data_n))

    def _save_error_data(self, val_err):
        """Save error data.
        Args:
            val_err(float): Validation error
        Returns:
            void
        """
        self._add_error(val_err, test_error=False)
        if self._use_test_data(): self._add_error(self.bestTestAcc, test_error=True)
    
    def _reset_unlabeled_data(self):
        """Remove accepted data from unlabeled data.
        Returns:
            X_un_init: updated unlabeled  data.
        """
        self.relative_indices = np.delete(self.relative_indices,
                                          np.asarray(self.keep_index),
                                          axis=0)
        self._init_index_to_keep()
        return self.X_un_init[self.relative_indices]
    def _if_no_data(self, number_of_data_points):
        """Check if unlabeled data is depleted.
        Returns:
            boolean
        """
        return number_of_data_points == 0

    def _init_fit(self, X_train, y_train, X_unlabeled):
        """Initiate everything that has to be initiated.
        Returns:
            void
        """
        (self._init_train(X_train, y_train), self._init_unlabeledData_relative_ix(X_unlabeled.shape[0]),
         self._init_index_to_keep(), self._init_unlabeled(X_unlabeled), self._init_model(X_train, y_train),
         self._save_model())
    
    def get_model(self):
        """Get model.
        Returns:
            model
        """
        return self.model
    
    def predict(self, X):
        """Make prediction.
        Returns:
            float: Prediction
        """
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """Make evaluation.
        Returns:
            float: evaluation
        """
        pred = np.around(self.model.predict(X))
        return accuracy_score(pred, y.ravel())
        
    def fit(self, X_train, y_train, X_test, y_test, X_unlabeled, X_val, y_val):
        """Begin fitting of semi-supervised.
        Args:
            X_train(arr): Training data
            y_train(arr): Training labels
            X_test(arr): Testing data
            y_test(arr): Testing labels
            X_unlabeled(arr): Unlabeled data
            X_val(arr): Valdiation data
            y_val(arr): Valdiation labels
        Returns:
            void
        """

        
        self._init_fit(X_train, y_train, X_unlabeled)

        if self.save_error:
            if np.any(X_test): self._init_save_error(test_error=True)
            self._init_save_error(test_error=False)
                
        
        if np.any(X_test):
            self.bestTestf1 = self.f1Test = self._get_f1_score(X_test, y_test)
            self.bestTestAcc = self.AccTest = self._get_acc_score(X_test, y_test)

            if self.save_error: self._add_error(self.bestTestAcc, test_error=True)
        else:
            self.bestTestAcc, self.AccTest = -1, -1

        bestValAcc = self._get_acc_score(X_val, y_val)
        
        
        if self.verbose: print("Init val error: ", bestValAcc)
            
        if self.save_error: self._add_error(bestValAcc, test_error=False)
    
        i = 0
        sample = False
        while self._if_train(i):

            newX, newy, ix = self._sample_new_data(X_unlabeled)
            
            if not sample:
                sampleX, sampley, tmp_sampleX, tmp_sampley = None, None, newX, newy
            else:
                tmp_sampleX, tmp_sampley = self._add_data(sampleX, sampley, newX, newy, sample=sample)
        
            X_train, y_train = self._add_data(tmp_sampleX, tmp_sampley)

            self._fit_model(X_train, y_train)

            

            if self._use_test_data():

                self.AccTest = self._get_acc_score(X_test, y_test)
                self.f1Test = self._get_f1_score(X_test, y_test)

            if self._if_better_model():
                sample = True
                sampleX, sampley = self._update_(tmp_sampleX, tmp_sampley, ix)
            else:
                if self.keras:
                    self._load_model()

            X_unlabeled = np.delete(X_unlabeled, ix, axis=0)

            if i % 1000 == 0:
                val_err = self._get_acc_score(X_val, y_val)
                self._save_error_data(val_err)
                if sample:
                    n_data = sampleX.shape[0]
                else:
                    n_data = 0
                if self.verbose: self._report(val_err, i, n_data)
                
                
                
                if self.keras:
                    self._load_model()
                    self.model.save_weights('assets/kerasmodels/model_weights_iter{}_val{}_test{}.h5'.format(i,
                                                                                                         np.round(val_err, 3),
                                                                                                         np.round(self.AccTest, 3)))

                np.save("assets/SemiSupArray/sampleX_iter{}_val{}_test{}".format(i,
                                                                               np.round(val_err, 3),
                                                                               np.round(self.AccTest, 3)),
                                                                               sampleX)

                np.save("assets/SemiSupArray/sampley_iter{}_val{}_test{}".format(i,
                                                                               np.round(val_err, 3),
                                                                               np.round(self.AccTest, 3)),
                                                                               sampley)
                
            i += 1

            if self._if_no_data(X_unlabeled.shape[0]):
                if self.verbose: print("One epoch of unlabeled data have passed; #new datapoints: ", len(self.keep_index))

                if self._if_no_data(len(self.keep_index)):
                    i = -1
                else:
                    X_unlabeled = self._reset_unlabeled_data()
                    if self._if_no_data(X_unlabeled.shape[0]):
                        i = -1
        print("Learning finnished")
            
     
