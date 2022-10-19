import numpy

class BinaryClassifier(object):
    """Base Binary Classifier

    Attributes:
        X (numpy.ndarray): Dataset of features.
        y (numpy.ndarray): The class labeling of the dataset calculated with clustering.
        cluster_labels_(numpy.ndarray): Set of labels existing in the dataset.
        cluster_mtx_(numpy.ndarray): Matrix of labels, each label gets its own array, that is filled with the actual label and the other labels but labeled as class 0. Shape: ( n_cluster, n_samples )
        models (sklearn Classification models): Sklearn Classifiers fitted as binary Classifiers.
    """
    X: numpy.ndarray
    y: numpy.ndarray
    cluster_labels_: numpy.ndarray
    cluster_mtx_: numpy.ndarray
    models: list
    model_proba: numpy.ndarray
    
    def __init__(self, X: numpy.ndarray, y: numpy.ndarray):
        """Constructor of base BinaryClassifier that takes 2 arguments.

        Args:
            X (numpy.ndarray): Dataset of shape ( n_samples, n_features ) 
            y (numpy.ndarray.): Labels of classes of the dataset with shape of ( n_samples )
        """
        self.X = X
        self.y = y
        self.__make_cluster_labels_()
        self.__make_cluster_mtx_()
    
    def __make_cluster_labels_(self):
        """Private method of Binary Classifier
        Fills cluster_labels_ attribute. This is the set of labels made from the list of labels.
        Each label exist once in this attribute.
        """
        self.cluster_labels_ = set(self.y)
    
    def __make_cluster_mtx_(self):
        """Private method of Binary Classifier
        This method fills the cluster_mtx_ attribute, that contains each label with their own dimension.
        """
        self.cluster_mtx_ = numpy.zeros_like((len(self.cluster_labels_), len(self.y)))
        for clr in self.cluster_labels_:
            for i in range(len(self.y)):
                if self.y[i] == clr:
                    self.cluster_mtx_[clr, i] = self.y[i]
                else:
                    self.cluster_mtx_[clr, i] = 0

class KNeighboursBinaryClassifier(BinaryClassifier):
    """KNeighbours binary classification.
    Derived class of Base binary classifier. 
    Attributes:
        models (list[Classifier]): The list of fitted binary classifiers.
        n_neighbours (int): Number of neighbors to use by default for kneighbors queries.
        n_jobs (int): Number of separate processed to be run. The higher the faster the training will be. Default is 16.
    """
    n_neighbours: int
    n_jobs: int

    def __init__(self, X, y, n_neighbours = 15, n_jobs = 16):
        """Constructor of KNeighboursBinaryClassifier, child of BinaryClassifier

        Args:
            X (numpy.ndarray): Dataset of shape ( n_samples, n_features ) 
            y (numpy.ndarray.): Labels of classes of the dataset with shape of ( n_samples )
            n_neighbours (int): Number of neighbors to use by default for kneighbors queries.
            n_jobs (int): Number of separate processed to be run. The higher the faster the training will be. Default is 16.
        """
        super().__init__(X, y)
        self.n_neighbours = n_neighbours
        self.n_jobs = n_jobs
    
    def fit(self):
        from sklearn.neighbors import KNeighborsClassifier
        for clr in self.cluster_labels_:
            self.models.append(KNeighborsClassifier(n_neighbors=self.n_neighbours, n_jobs=self.n_jobs).fit(self.X, self.cluster_mtx_[clr]))