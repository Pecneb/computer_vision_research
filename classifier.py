import numpy
from sklearn.base import ClassifierMixin

class BinaryClassifier(ClassifierMixin):
    """Base Binary Classifier

    Attributes:
        X (numpy.ndarray): Dataset of features.
        y (numpy.ndarray): The class labeling of the dataset calculated with clustering.
        class_labels_(numpy.ndarray): Set of labels existing in the dataset.
        label_mtx_(numpy.ndarray): Matrix of labels, each label gets its own array, that is filled with the actual label and the other labels but labeled as class 0. Shape: ( n_cluster, n_samples )
        models (sklearn Classification models): Sklearn Classifiers fitted as binary Classifiers.
        class_proba_ (numpy.ndarray): Array of probabilities of the predictions made by the binary classifiers. Each i-th element in the array corresponds to the class number i. This array holds the most recent prediction's probabilities.
    """
    X: numpy.ndarray
    y: numpy.ndarray
    class_labels_: numpy.ndarray
    label_mtx_: numpy.ndarray
    models_: list
    class_proba_: numpy.ndarray
    
    def __init__(self, X: numpy.ndarray, y: numpy.ndarray):
        """Constructor of base BinaryClassifier that takes 2 arguments.

        Args:
            X (numpy.ndarray): Dataset of shape ( n_samples, n_features ) 
            y (numpy.ndarray.): Labels of classes of the dataset with shape of ( n_samples )
        """
        self.X = X
        self.y = y
        self.__make_class_labels_()
        self.__make_label_mtx_()
    
    def __make_class_labels_(self):
        """Private method of Binary Classifier
        Fills class_labels_ attribute. This is the set of labels made from the list of labels.
        Each label exist once in this attribute.
        """
        self.class_labels_ = numpy.array(list(set(self.y)), dtype=int).reshape((1, len(set(self.y))))
    
    def __make_label_mtx_(self):
        """Private method of Binary Classifier
        This method fills the label_mtx_ attribute, that contains each label with their own dimension.
        """
        self.label_mtx_= numpy.zeros((self.class_labels_.shape[1], self.y.shape[0]), dtype=int)
        for clr in self.class_labels_[0]:
            for i in range(len(self.y)):
                if self.y[i] == clr:
                    self.label_mtx_[clr, i] = 1
                else:
                    self.label_mtx_[clr, i] = 0
    
    def init_models(self, classifier: ClassifierMixin, **classifier_args):
        """Initialize self.models_ with sklearn classifiers

        Args:
            classifier (ClassifierMixin): The classifier that should be used for binary classification.
        """
        self.models_ = []
        for clr in self.class_labels_[0]:
            self.models_.append(classifier(**classifier_args))
    
    def fit(self):
        """Fit sklearn classifier models with given dataset given at initialization.
        """
        for clr, mdl in zip(self.class_labels_[0], self.models_):
            mdl.fit(self.X, self.label_mtx_[clr])
    
    def predict_proba(self, X: numpy.ndarray):
        """Return predicted probabilities of dataset X

        Args:
            X (numpy.ndarray): Feature vector of shape( n_samples, n_features ) for prediction. 

        Returns:
            numpy.ndarray: Prediction probabilities of shape( n_samples, n_classlabels ) 
        """
        self.class_proba_ = numpy.zeros((X.shape[0], self.class_labels_.shape[1]))
        for clr, mdl in zip(self.class_labels_[0], self.models_):
            self.class_proba_[:, clr] = mdl.predict_proba(X)[:, 1]
        return self.class_proba_
    
    #TODO validation on each class
    def validate(self, X_test: numpy.ndarray, y_test: numpy.ndarray, threshold: numpy.float32):
        """Validate trained models.

        Args:
            X_test (numpy.ndarray): Validation dataset of shape( n_samples, n_features ). 
            y_test (numpy.ndarray): Validation class labels shape( n_samples, 1 ). 
            threshold (numpy.float32): Probability threshold, if prediction probability higher than the threshold, then it counts as a valid prediction.
        """
        predict_proba_results = self.predict_proba(X_test)
        tt = 0
        tf = 0
        ff = 0
        ft = 0
        for i in range(y_test.shape[0]):
            if predict_proba_results[i, y_test[i]] >= threshold:
                
        return (hit / y_test.shape[0])