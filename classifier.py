from joblib import Parallel, delayed
import numpy
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import validation
from sklearn import multiclass
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multiclass import check_is_fitted

class BinaryClassifier(ClassifierMixin, BaseEstimator):
    """Base Binary Classifier

    A classifier that can take any type of scikit-learn classifier as the binary classifier. 
    At the creation of the classifier object, a scikit learn classifier have to be given
    as the first argument, the second argument is a dictionary of args that will be passed
    to the scikit learn classifier.

    Parameters
    ----------
    classifier : scikit-learn classifier
        A scikit-learn classifier that will be used to create the multiclass binary classifier. 

    Attributes
    ----------
         Attributes
    ----------
    X_ : ndarray, shape (n_samples, n_features)
        The input passed during :meth:`fit`.
    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.
    label_mtx_ : numpy.ndarray shape (n_cluster, n_samples) 
        Matrix of labels, each label gets its own array, 
        that is filled with the actual label and the other labels but labeled as class 0. 
    models : scikit-learn classifier
        Sklearn Classifiers fitted as binary Classifiers.
    """
    #trackData: list 
    #X_: numpy.ndarray
    #y_: numpy.ndarray
    #class_labels_: numpy.ndarray
    #label_mtx_: numpy.ndarray
    #models_: list

    
    def __init__(self, trackData: list, classifier: ClassifierMixin, classifier_argv: dict = None):
        """Constructor of the BinaryClassifier.

        Args:
            trackData : list[TrackedObject]
                List of object track's
            classifier : scikit-learn classifier 
                This scikit-learn classifier class will be used as binary classifier.
            classifier_argv : dict
                These arguments will be passed to the given scikit-learn classifier.
        """
        self.trackData = trackData
        self.classifier = classifier
        self.classifier_argv = classifier_argv
    
    def __make_class_labels_(self):
        """Private method of Binary Classifier
        Fills class_labels_ attribute. This is the set of labels made from the list of labels.
        Each label exist once in this attribute.
        """
        self.classes_ = numpy.array(list(set(self.y)), dtype=int).reshape((1, len(set(self.y))))
    
    def __make_label_mtx_(self):
        """Private method of Binary Classifier

        Creates binary label matrix.
        class 0 [[0,1,1,0], if label is class 0 then 1, 0 otherwise
        class 1  [1,1,0,1], if label is class 1 then 1, 0 otherwise
        class 2  [0,1,0,0]] if label is class 2 then 1, 0 otherwise
        Every row corresponds to its class label, and every column is a label, that can be 1 or 0.
        """
        self.label_mtx_ = numpy.zeros((self.classes_.shape[0], self.y_.shape[0]), dtype=int)
        for i in range(self.classes_.shape[0]):
            for j in range(self.y_.shape[0]):
                if self.y_[j] == self.classes_[i]:
                    self.label_mtx_[i, j] = 1
                else:
                    self.label_mtx_[i, j] = 0
    
    def __init_models(self):
        """Initialize self.models_ with sklearn classifiers
        """
        self.models_ = []
        for clr in self.classes_:
            if self.classifier_argv is not None:
                self.models_.append(self.classifier().set_params(**self.classifier_argv))
            else:
                self.models_.append(self.classifier())
    
    def fit(self, X: numpy.ndarray, y: numpy.ndarray):
        """Fit sklearn classifier models with given dataset X and class labels y.

        Parameters
        ----------
        X : numpy.ndarray shape (n_samples, n_features)
            The training input samples.
        y : numpy.ndarray shape (n_samples,)
            The target values. An array of int. (labels)

        Returns
        -------
        self : object
            Returns self. Fitted model.
        """

        #if X is None and y is None:
        #    for clr, mdl in zip(self.class_labels_[0], self.models_):
        #        mdl.fit(self.X, self.label_mtx_[clr])
        #else:

        # Check that X and y have correct shape
        X, y = validation.check_X_y(X, y) 
        # Store the classes seen during fit
        self.classes_ = multiclass.unique_labels(y)
        # Class labels can only be integers
        self.classes_ = self.classes_.astype(int)
        
        self.X_ = X
        self.y_ = y
        
        # Create label matrix shape (n_classes, n_samples)
        self.__make_label_mtx_()
        
        self.__init_models()

        # Fit every classifier
        for clr, mdl in zip(range(self.classes_.shape[0]), self.models_):
            mdl.fit(X, self.label_mtx_[clr])

        # Return fitted classifier
        return self
    
    def predict_proba(self, X: numpy.ndarray):
        """Return predicted probabilities of dataset X

        Parameters
        ----------
        X : numpy.ndarray shape (n_samples, n_features)

        Returns:
        numpy.ndarray : shape (n_samples, n_classes_) 
            Prediction probabilities of 
        """
        X = validation.check_array(X, ensure_2d=True)
        class_proba = numpy.zeros((X.shape[0], self.classes_.shape[0]))
        for clr, mdl in zip(range(self.classes_.shape[0]), self.models_):
            class_proba[:, clr] = mdl.predict_proba(X)[:, 1]
        return class_proba

    def predict(self, X: numpy.ndarray,  threshold: numpy.float32 = 0.5, top:int=1):
        """Return predicted top labels of dataset  X

        Args:
            X (numpy.ndarray): Feature vector of shape( n_samples, n_features ) for prediction. 
            top (int): Number tof classes with the highest probability

        Returns:
            numpy.ndarray: lists of prediction result class labels, lenght=top
        """
        if top > len(self.classes_):
            print("PARAMETER ERROR: The value of TOP must be lower or equal than the number of classes")
        class_proba = self.predict_proba(X=X)
        # print(class_proba)
        #print(class_proba.shape)
        prediction_result = numpy.argsort(class_proba)
        # print(prediction_result)
        top_pred_res = numpy.zeros(prediction_result.shape)
        """for i, sor in enumerate(prediction_result):
            for oszlop in sor:
                if self.class_proba_[i,oszlop] < threshold:
                    top_pred_res[i,oszlop] = -1
                else:
                    top_pred_res[i,oszlop] = prediction_result[i,oszlop]
        """
        # print(prediction_result[:,-top:])
        return prediction_result[:,-top:]
        #return top_pred_res[:,-top:]

    
    #TODO validation on each class
    def validate(self, X_test: numpy.ndarray, y_test: numpy.ndarray, threshold: numpy.float32):
        """Validate trained models.
        Args:
            X_test (numpy.ndarray): Validation dataset of shape( n_samples, n_features ). 
            y_test (numpy.ndarray): Validation class labels shape( n_samples, 1 ). 
            threshold (numpy.float32): Probability threshold, if prediction probability higher than the threshold, then it counts as a valid prediction.
        """
        predict_proba_results = self.predict_proba(X_test)
        accuracy_vector = []
        balanced_accuracy = []
        for j in self.classes_: 
            tp = 0 # True positive --> predicting true and it is really true 
            fn = 0 # False negative (type II error) --> predicting false, although its true 
            tn = 0 # True negative --> predicting false and it is really false
            fp = 0 # False positive (type I error) --> predicting true, although it is false
            for i, _y_test in enumerate(y_test):
                if _y_test == j:
                    if predict_proba_results[i, j]>= threshold:
                        tp += 1
                    else:
                        fn += 1
                else: 
                    if predict_proba_results[i, j]< threshold:
                        tn += 1
                    else:
                        fp += 1
            if tp + fn == 0:
                sens = 0
            else:
                sens = tp/(tp+fn)
            if (tn + fp) == 0:
                spec = 0
            else:
                spec = tn/(tn+fp)
            balanc = (sens + spec) / 2
            tp = tp/len(y_test)
            tn = tn/len(y_test)
            fn = fn/len(y_test)
            fp = fp/len(y_test) 
            accuracy_vector.append(tp+tn)
            balanced_accuracy.append(balanc)
        return balanced_accuracy

    def validate_predictions(self, X_test: numpy.ndarray, y_test: numpy.ndarray, threshold: numpy.float32, top: int=1):
        """Validate trained models.
        Args:
            X_test (numpy.ndarray): Validation dataset of shape( n_samples, n_features ). 
            y_test (numpy.ndarray): Validation class labels shape( n_samples, 1 ). 
            threshold (numpy.float32): Probability threshold, if prediction probability higher than the threshold, then it counts as a valid prediction.
        """
        predict_results = self.predict(X_test, threshold=threshold, top=top)
        #print(predict_results)
        #print(predict_results.shape)
        accuracy_vector = []
        balanced_accuracy = []
        tp = 0 # True positive --> predicting true and it is really true 
        fn = 0 # False negative (type II error) --> predicting false, although its true 
        tn = 0 # True negative --> predicting false and it is really false
        fp = 0 # False positive (type I error) --> predicting true, although it is false
        for i, _y_test in enumerate(y_test):
            # print(_y_test in predict_results[i])
            if _y_test in predict_results[i]:
                tp += 1
            else:
                fp +=1
        #print(tp, len(y_test))
        map_ = tp/len(y_test)
        #print(map_)
        return map_


class OneVSRestClassifierExtended(OneVsRestClassifier):
    """Extended One vs. Rest Classifier

    The predict() method takes 2 extra agruments, threshold and a top n.
    Only the top n classes over the threshold will be returned.
    Also a validate and a validate_predictions method is implemented.
    The validate() method calculates the balanced accuracy of the classifier.
    It takes 3 arguments, X_test, y_test, threshold. X and y test are the
    testing datasets, the threshold is a float value that 

    Attributes
    ----------

    Parameters
    ----------
    estimator : estimator object
        A regressor or a classifier that implements fit. When a classifier is passed, 
        decision_function will be used in priority and it will fallback to :term`predict_proba` 
        if it is not available. When a regressor is passed, predict is used.
    tracks : list
    
    """

    def __init__(self, estimator, tracks, n_jobs=16):
        """_summary_

        Parameters
        ----------
        estimator : classifier 
            Scikit-Learn classifier object, that will be used
            to create the binary classifiers.
        tracks : list[TrackedObject]
            Filtered track dataset, that was used to create
            feature vectors for clustering, and X, y feature
            vectors and labels. 
        n_jobs : int
            Number of processes to run. 
            Default n_jobs value is 16.
        """
        self.tracks = tracks
        super().__init__(estimator, n_jobs=n_jobs)
    
    def fit(self, X, y, centroids: dict = None):
        """Fit underlying estimators.
        If centroids dictionary are given,
        then version three feature vectors
        are calculated per cluster.

        Parameters
        ----------
        X : (sparse) array-like of shape (n_samples, n_features)
            Data.

        y : (sparse) array-like of shape (n_samples,) or (n_samples, n_classes)
            Multi-class targets. An indicator matrix turns on multilabel
            classification.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        # A sparse LabelBinarizer, with sparse_output=True, has been shown to
        # outperform or match a dense label binarizer in all cases and has also
        # resulted in less or equal memory consumption in the fit_ovr function
        # overall.
        self.label_binarizer_ = LabelBinarizer(sparse_output=True)
        Y = self.label_binarizer_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_binarizer_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        # In cases where individual estimators are very fast to train setting
        # n_jobs > 1 in can results in slower performance due to the overhead
        # of spawning threads.  See joblib issue #112.
        if centroids is not None:
            self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(multiclass._fit_binary)(
                    self.estimator,
                    numpy.array([numpy.append(x, [centroids[i][0] - x[0], 
                                            centroids[i][1] - x[1], 
                                            centroids[i][0] - x[2], 
                                            centroids[i][1] - x[3], 
                                            centroids[i][0] - x[4], 
                                            centroids[i][1] - x[5]]) for x in X]),
                    column,
                    classes=[
                        "not %s" % self.label_binarizer_.classes_[i],
                        self.label_binarizer_.classes_[i],
                    ],
                )
                for i, column in enumerate(columns)
            )
        else:
            self.estimators_ = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(multiclass._fit_binary)(
                    self.estimator,
                    X,
                    column,
                    classes=[
                        "not %s" % self.label_binarizer_.classes_[i],
                        self.label_binarizer_.classes_[i],
                    ],
                )
                for i, column in enumerate(columns)
            )

        if hasattr(self.estimators_[0], "n_features_in_"):
            self.n_features_in_ = self.estimators_[0].n_features_in_
        if hasattr(self.estimators_[0], "feature_names_in_"):
            self.feature_names_in_ = self.estimators_[0].feature_names_in_

        return self

       
    def predict_proba(self, X: numpy.ndarray, centroids: dict = None):
        """Return predicted probabilities of dataset X.
        If cluster centroids dictionary is given, then 
        version three feature vectors are used, that are
        calculated per cluster.

        Parameters
        ----------
        X : numpy.ndarray shape (n_samples, n_features)

        Returns:
        numpy.ndarray : shape (n_samples, n_classes_) 
            Prediction probabilities of 
        """
        check_is_fitted(self)
        X = validation.check_array(X, ensure_2d=True)
        

        Y = numpy.zeros((X.shape[0], self.classes_.shape[0]))
        for clr, mdl in zip(range(self.classes_.shape[0]), self.estimators_):
            if centroids is not None:
                Y[:, clr] = mdl.predict_proba(numpy.array([numpy.append(x, [
                    centroids[clr][0] - x[0], 
                    centroids[clr][1] - x[1], 
                    centroids[clr][0] - x[2], 
                    centroids[clr][1] - x[3], 
                    centroids[clr][0] - x[4], 
                    centroids[clr][1] - x[5]
                ]) for x in X]))[:, 1]
            else:
                Y[:, clr] = mdl.predict_proba(X)[:, 1]
        return Y 

    def predict(self, X: numpy.ndarray, top:int=1, centroids: dict = None):
        """Return predicted top labels of dataset X

        Parameters
        ----------
        X : numpy.ndarray shape (n_samples, n_features)
            Sample dataset.
        threshold : numpy.float32
            Probability threshold, if prediction is above this value,
            then it has a chance to get in the top n predictions.
        top : int
            Number of classes with the highest probability

        Returns
        -------
        predictions : numpy.ndarray shape (top,) 
            Top n classes.
        """
        check_is_fitted(self)
        X = validation.check_array(X, ensure_2d=True)

        if top > len(self.classes_):
            print("PARAMETER ERROR: The value of TOP must be lower or equal than the number of classes")

        # Get probability for all classes.
        if centroids is not None:
            class_proba = self.predict_proba(X=X, centroids=centroids)
        else:
            class_proba = self.predict_proba(X=X)
        # Sort to ascending order.
        prediction_result = numpy.argsort(class_proba)
        
        #top_pred_res = numpy.zeros(prediction_result.shape)
        #print(prediction_result[:,-top:])
        return prediction_result[:,-top:]
        #return top_pred_res[:,-top:]

    def validate(self, X_test: numpy.ndarray, y_test: numpy.ndarray, threshold: numpy.float32, centroids: dict = None):
        """Validate trained models.

        Calculates the balanced accuracy of the underlying trained models.

        Parameters
        ----------
        X_test : numpy.ndarray shape ( n_samples, n_features )
            Test dataset. 
        y_test : numpy.ndarray shape (n_samples,)
            Labels of the test dataset.
        threshold : numpy.float32 
            Probability threshold, if prediction probability higher
            than the threshold, then it counts as a valid prediction.

        Returns
        -------
        balanced_accuracy : float, shape (n_classes,)
        """
        if centroids is not None:
            predict_proba_results = self.predict_proba(X_test, centroids)
        else:
            predict_proba_results = self.predict_proba(X_test)
        accuracy_vector = []
        balanced_accuracy = []
        for j in self.classes_: 
            tp = 0 # True positive --> predicting true and it is really true 
            fn = 0 # False negative (type II error) --> predicting false, although its true 
            tn = 0 # True negative --> predicting false and it is really false
            fp = 0 # False positive (type I error) --> predicting true, although it is false
            for i, _y_test in enumerate(y_test):
                if _y_test == j:
                    if predict_proba_results[i, j]>= threshold:
                        tp += 1
                    else:
                        fn += 1
                else: 
                    if predict_proba_results[i, j]< threshold:
                        tn += 1
                    else:
                        fp += 1
            if tp + fn == 0:
                sens = 0
            else:
                sens = tp/(tp+fn)
            if (tn + fp) == 0:
                spec = 0
            else:
                spec = tn/(tn+fp)
            balanc = (sens + spec) / 2
            tp = tp/len(y_test)
            tn = tn/len(y_test)
            fn = fn/len(y_test)
            fp = fp/len(y_test) 
            accuracy_vector.append(tp+tn)
            balanced_accuracy.append(balanc)
        return balanced_accuracy

    def validate_predictions(self, X_test: numpy.ndarray, y_test: numpy.ndarray, top: int=1, centroids: dict = None):
        """Validate trained models.

        Parameters
        ----------
        X_test : numpy.ndarray shape ( n_samples, n_features )
            Test dataset. 
        y_test : numpy.ndarray shape (n_samples,)
            Labels of the test dataset.
        threshold : numpy.float32 
            Probability threshold, if prediction probability higher
            than the threshold, then it counts as a valid prediction.
        top : int
            Parameter for predict() method, that returns the top n predictions.
        """
        if centroids is not None:
            predict_results = self.predict(X_test, top=top, centroids=centroids)
        else:
            predict_results = self.predict(X_test, top=top)

        tp = 0 # True positive --> predicting true and it is really true 
        fn = 0 # False negative (type II error) --> predicting false, although its true 
        tn = 0 # True negative --> predicting false and it is really false
        fp = 0 # False positive (type I error) --> predicting true, although it is false

        for i, _y_test in enumerate(y_test):
            if _y_test in predict_results[i]:
                tp += 1
            else:
                fp +=1

        map_ = tp/len(y_test)

        return map_