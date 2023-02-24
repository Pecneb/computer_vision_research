"""
    Predicting trajectories of objects
    Copyright (C) 2022  Bence Peter

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Contact email: ecneb2000@gmail.com
"""
from datetime import date 
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from processing_utils import (
    load_model, 
    save_model, 
    data_preprocessing_for_calibrated_classifier, 
    data_preprocessing_for_classifier, 
    data_preprocessing_for_classifier_from_joblib_model, 
    preprocess_dataset_for_training,
    load_joblib_tracks
    )
from tqdm import tqdm
np.seterr(divide='ignore', invalid='ignore')

def KNNClassification(X: np.ndarray, y: np.ndarray, n_neighbours: int):
    """Run K Nearest Neighbours classification on samples X and labels y with neighbour numbers n_neighbours.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels
        n_neighbours (int): Number of neighbours to belong in a class.

    Returns:
        sklearn classifier: KNN model
    """
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=n_neighbours, weights='distance').fit(X, y)
    return classifier

def SGDClassification(X: np.ndarray, y: np.ndarray):
    """Run Stochastic Gradient Descent Classification on samples X and labels y.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: SGD model 
    """
    from sklearn.linear_model import SGDClassifier
    classifier = SGDClassifier(loss="modified_huber").fit(X, y)
    return classifier

def GPClassification(X: np.ndarray, y: np.ndarray):
    """Run Gaussian Process Classification on samples X and labels y.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: GP model 
    """
    from sklearn.gaussian_process import GaussianProcessClassifier
    classifier = GaussianProcessClassifier().fit(X, y)
    return classifier

def GNBClassification(X: np.ndarray, y: np.ndarray):
    """Run Gaussian Naive Bayes Classification on samples X and labels y.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: GNB model 
    """
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB().fit(X, y)
    return classifier

def MLPClassification(X: np.ndarray, y: np.ndarray):
    """Run Multi Layer Perceptron Classification on samples X and labels y.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: MLPC model 
    """
    from sklearn.neural_network import MLPClassifier
    classifier = MLPClassifier(max_iter=1000).fit(X,y)
    return classifier

def VotingClassification(X: np.ndarray, y: np.ndarray):
    """Run Voting Classification on samples X and labels y.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: Voting model 
    """
    from sklearn.ensemble import VotingClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    clf1 = KNeighborsClassifier(n_neighbors=15, weights='distance')
    clf2 = SGDClassifier()
    clf3 = GaussianProcessClassifier()
    clf4 = GaussianNB()
    clf5 = MLPClassifier()
    classifier = VotingClassifier(
        estimators=[('knn', clf1), ('sgd', clf2), ('gp', clf3), ('gnb', clf4), ('mlp', clf5)]
    ).fit(X, y)
    return classifier

def SVMClassficitaion(X: np.ndarray, y: np.ndarray):
    """Run Support Vector Machine classification with RBF kernel.

    Args:
        X (np.ndarray): Dataset
        y (np.ndarray): labels

    Returns:
        skelarn classifier: SVM model
    """
    from sklearn.svm import SVC
    classifier = SVC().fit(X, y)
    return classifier

def DTClassification(X: np.ndarray, y: np.ndarray):
    """Run decision tree classification.

    Args:
        X (np.ndarray): Dataset 
        y (np.ndarray): labels
    """
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier().fit(X, y)
    return classifier

def Classification(classifier: str, path2db: str, **argv):
    """Run classification on database data.

    Args:
        classifier (str): Type of the classifier. 
        path2db (str): Path to database file. 

    Returns:
        bool: Returns false if bad classifier was given. 
    """
    X_train, y_train, _, X_valid, y_valid, _, _ = data_preprocessing_for_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'])
    fig, ax = plt.subplots()
    model = None
    if classifier == 'KNN':
        model = KNNClassification(X_train, y_train, 15)
    elif classifier == 'SGD':
        model = SGDClassification(X_train, y_train)
    elif classifier == 'GP':
        model = GPClassification(X_train, y_train)
    elif classifier == 'GNB':
        model = GNBClassification(X_train, y_train)
    elif classifier == 'MLP':
        model = MLPClassification(X_train, y_train)
    elif classifier == 'VOTE':
        model = VotingClassification(X_train, y_train)
    elif classifier == 'SVM':
        model = SVMClassficitaion(X_train, y_train)
    elif classifier == 'DT':
        model = DTClassification(X_train, y_train)
    else:
        print(f"Error: bad classifier {classifier}")
        return False
    ValidateClassification(model, X_valid, y_valid)
    """xx, yy= np.meshgrid(np.arange(0, 2, 0.005), np.arange(0, 2, 0.005))
    X_visualize = np.zeros(shape=(xx.shape[0]*xx.shape[1],6))
    counter = 0
    for i in range(0,xx.shape[0]):
        for j in range(0,xx.shape[1]):
            X_visualize[counter,0] = xx[j,i]
            X_visualize[counter,1] = yy[j,i]
            X_visualize[counter,2] = xx[j,i]
            X_visualize[counter,3] = yy[j,i]
            X_visualize[counter,4] = xx[j,i]
            X_visualize[counter,5] = yy[j,i]
            counter += 1
    y_visualize = model.predict(X_visualize)
    ax.pcolormesh(xx,yy,y_visualize.reshape(xx.shape))
    ax.scatter(X_train[:, 0], 1-X_train[:, 1], c=y_train, edgecolors='k')
    ax.set_ylim(0,2)
    ax.set_xlim(0,2)
    plt.show()"""
    save_model(path2db, str("model_"+classifier), model)

def ClassificationWorker(path2db: str, **argv):
    """Run all of the classification methods implemented.

    Args:
        path2db (str): Path to database file. 
    """
    X_train, y_train, _, X_valid, y_valid, _, _= data_preprocessing_for_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'])
    print("KNN")
    model = KNNClassification(X_train, y_train, 15)
    ValidateClassification(model, X_valid, y_valid)
    print("SGD")
    model = SGDClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid) 
    print("GP")
    model = GPClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)
    print("GNB")
    model = GNBClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)
    print("MLP")
    model = MLPClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)
    print("VOTE")
    model = VotingClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)
    print("SVM")
    model = SVMClassficitaion(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)
    print("DT")
    model = DTClassification(X_train, y_train)
    ValidateClassification(model, X_valid, y_valid)

def ValidateClassification(clfmodel, X_valid: np.ndarray, y_valid: np.ndarray):
    """Validate fitted classification model.

    Args:
        clfmodel (str, model): Can be a path to model file or model. 
        X_valid (np.ndarray): Test dataset. 
        y_valid (np.ndarray): Test dataset's labeling. 
    """
    if type(clfmodel) is str:
        model = joblib.load(clfmodel)
    else:
        model = clfmodel
    y_predict = model.predict(X_valid)
    assert len(y_predict) == len(y_valid)
    print(f"Number of mislabeled points out of a total {X_valid.shape[0]} points : {(y_valid != y_predict).sum()} \nAccuracy: {(1-((y_valid != y_predict).sum() / X_valid.shape[0])) * 100} %")

def ValidateClassification_Probability(clfmodel, X_valid: np.ndarray, y_valid: np.ndarray, threshold: np.float64):
    """Calculate accuracy of classification model using the predict_proba method of the classifier.

    Args:
        clfmodel (str, model): Can be a path to model file or model. 
        X_valid (np.ndarray): Test dataset. 
        y_valid (np.ndarray):  Test dataset's labeling.
    """
    if type(clfmodel) is str:
        model = joblib.load(clfmodel)
    else:
        model = clfmodel
    y_predict_proba = model.predict_proba(X_valid)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i, y_proba_vec in enumerate(y_predict_proba):
        for j, y_proba in enumerate(y_proba_vec):
            if j == y_valid[i]:
                if y_proba >= threshold:
                    tp += 1 
                else:
                    fn += 1
            else:
                if y_proba < threshold:
                    tn += 1
                else:
                    fp += 1
    return True
                
def CalibratedClassification(classifier: str, path2db: str, **argv):
    """Run classification on database data.

    Args:
        classifier (str): Type of the classifier. 
        path2db (str): Path to database file. 

    Returns:
        bool: Returns false if bad classifier was given. 
    """
    from sklearn.calibration import CalibratedClassifierCV
    X_train, y_train, X_calib, y_calib, X_valid, y_valid = data_preprocessing_for_calibrated_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'])
    fig, ax = plt.subplots()
    model = None
    if classifier == 'KNN':
        model = KNNClassification(X_train, y_train, 15)
    elif classifier == 'SGD':
        model = SGDClassification(X_train, y_train)
    elif classifier == 'GP':
        model = GPClassification(X_train, y_train)
    elif classifier == 'GNB':
        model = GNBClassification(X_train, y_train)
    elif classifier == 'MLP':
        model = MLPClassification(X_train, y_train)
    elif classifier == 'VOTE':
        model = VotingClassification(X_train, y_train)
    else:
        print(f"Error: bad classifier {classifier}")
        return False
    model_calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    model_calibrated.fit(X_calib, y_calib)
    ValidateClassification(model_calibrated, X_valid, y_valid)
    xx, yy= np.meshgrid(np.arange(0, 2, 0.005), np.arange(0, 2, 0.005))
    X_visualize = np.zeros(shape=(xx.shape[0]*xx.shape[1],6))
    counter = 0
    for i in range(0,xx.shape[0]):
        for j in range(0,xx.shape[1]):
            X_visualize[counter,0] = xx[j,i]
            X_visualize[counter,1] = yy[j,i]
            X_visualize[counter,2] = xx[j,i]
            X_visualize[counter,3] = yy[j,i]
            X_visualize[counter,4] = xx[j,i]
            X_visualize[counter,5] = yy[j,i]
            counter += 1
    y_visualize = model.predict(X_visualize)
    ax.pcolormesh(xx,yy,y_visualize.reshape(xx.shape))
    ax.scatter(X_train[:, 0], 1-X_train[:, 1], c=y_train, edgecolors='k')
    ax.set_ylim(0,2)
    ax.set_xlim(0,2)
    plt.show()
    save_model(path2db, str("calibrated_model_"+classifier), model_calibrated)

def CalibratedClassificationWorker(path2db: str, **argv):
    """Run all the classification methods implemented.

    Args:
        path2db (str): Path to database file. 
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    X_train, y_train, _, X_valid, y_valid, _, _ = data_preprocessing_for_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'])
    #vote = VotingClassification(X_train, y_train)
    models = {
        'KNN' : KNeighborsClassifier(n_neighbors=15),
        'SGD' : SGDClassifier(),
        'GP' : GaussianProcessClassifier(n_jobs=argv['n_jobs']),
        'GNB' : GaussianNB(),
        'MLP' : MLPClassifier()
    }
    for cls in models:
        print(cls)
        calibrated = CalibratedClassifierCV(models[cls], method="sigmoid", n_jobs=18).fit(X_train, y_train)
        ValidateClassification(calibrated, X_valid, y_valid)

def BinaryClassificationWorkerTrain(path2db: str, path2model = None, **argv):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from classifier import OneVSRestClassifierExtended
    from sklearn.tree import DecisionTreeClassifier
    from processing_utils import strfy_dict_params

    X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid, tracks = [], [], [], [], [], [], []

    if path2model is not None:
        model = load_model(path2model)
        tracks = model.tracks
        X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid = data_preprocessing_for_classifier_from_joblib_model(model, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'], 
                                                            from_half=argv['from_half'],
                                                            features_v2=argv['features_v2'],
                                                            features_v2_half=argv['features_v2_half'],
                                                            features_v3=argv['features_v3'])
    else:
        X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid, tracks = data_preprocessing_for_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'], from_half=argv['from_half'],
                                                            features_v2=argv['features_v2'],
                                                            features_v2_half=argv['features_v2_half'],
                                                            features_v3=argv['features_v3'])

    """X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid, tracks, cluster_centroids = preprocess_dataset_for_training(
        path2dataset=path2db, 
        min_samples=argv['min_samples'], 
        max_eps=argv['max_eps'], 
        xi=argv['xi'], 
        min_cluster_size=argv['min_cluster_size'],
        n_jobs=argv['n_jobs'], 
        from_half=argv['from_half'],
        features_v2=argv['features_v2'],
        features_v2_half=argv['features_v2_half'],
        features_v3=argv['features_v3']
    )"""

    models = {
        'KNN' : KNeighborsClassifier,
        'GP' : GaussianProcessClassifier,
        'GNB' : GaussianNB,
        'MLP' : MLPClassifier,
        'SGD' : SGDClassifier,
        'SVM' : SVC,
        'DT' : DecisionTreeClassifier
    }
    
    parameters = {
        'KNN' : {'n_neighbors' : 15},
        'GP' :  {},
        'GNB' : {},
        'MLP' : {'max_iter' : 1000, 'solver' : 'sgd'},
        'SGD' : {'loss' : 'modified_huber'},
        'SVM' : {'kernel' : 'rbf', 'probability' : True},
        'DT' : {} 
    }

    table = pd.DataFrame()
    table2 = pd.DataFrame()
    probability_over_time = pd.DataFrame()

    if not os.path.isdir(os.path.join('research_data', path2db.split('/')[-1].split('.')[0], "tables")):
            os.mkdir(os.path.join('research_data', path2db.split('/')[-1].split('.')[0], "tables"))
    savepath = os.path.join(os.path.join('research_data', path2db.split('/')[-1].split('.')[0], "tables"))

    for clr in tqdm(models, desc="Classifier trained."):
        binaryModel = OneVSRestClassifierExtended(models[clr](**parameters[clr]), tracks, n_jobs=argv['n_jobs'])
        #binaryModel = BinaryClassifier(trackData=tracks, classifier=models[clr], classifier_argv=parameters[clr])
        #binaryModel.init_models(models[clr])

        binaryModel.fit(X_train, y_train)

        top_picks = []
        for i in range(1,4):
            top_picks.append(binaryModel.validate_predictions(X_valid, y_valid, top=i))
        balanced_threshold = binaryModel.validate(X_valid, y_valid, argv['threshold'])

        table[clr] = np.asarray(top_picks)
        table2[clr] = balanced_threshold

        probabilities = binaryModel.predict_proba(X_valid)
        for i in range(probabilities.shape[1]):
            probability_over_time[f"Class {i}"] = probabilities[:, i]
        probability_over_time["Time_Enter"] = metadata_valid[:, 0]
        probability_over_time["Time_Mid"] = metadata_valid[:, 1]
        probability_over_time["Time_Exit"] = metadata_valid[:, 2]
        probability_over_time["History_Length"] = metadata_valid[:, 3]
        probability_over_time["TrackID"] = metadata_valid[:, 4]
        probability_over_time["True_Class"] = y_valid 

        filename = os.path.join(savepath, f"{date.today()}_{clr}.xlsx")
        with pd.ExcelWriter(filename) as writer:
            probability_over_time.to_excel(writer, sheet_name="Probability_over_time") # each feature vector
            table.to_excel(writer, sheet_name="Top_Picks") # top n accuracy
            table2.to_excel(writer, sheet_name="Balanced") # balanced accuracy

        #TODO: somehow show in title which feature vectors were used for the tarining 
        if argv['from_half']:
            save_model(path2db, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_from_half"), binaryModel) 
        elif argv['features_v2']:
            save_model(path2db, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v2"), binaryModel)
        elif argv['features_v2_half']:
            save_model(path2db, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v2_from_half"), binaryModel)
        elif argv['features_v3']:
            save_model(path2db, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v3"), binaryModel)


    table.index += 1
    print("Top picks")
    print(table.to_markdown())
    print("Threshold")
    print(table2.to_markdown())
    print(table2.aggregate(np.average).to_markdown())

def train_binary_classifiers(path2dataset: str, outdir: str, **argv):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from classifier import OneVSRestClassifierExtended
    from sklearn.tree import DecisionTreeClassifier
    from processing_utils import strfy_dict_params, iter_minibatches
    from visualizer import aoiextraction

    if argv['classification_features_version'] == "v4":
        X, y, metadata, tracks, labels = preprocess_dataset_for_training(
            path2dataset=path2dataset, 
            min_samples=argv['min_samples'], 
            max_eps=argv['max_eps'], 
            xi=argv['xi'], 
            min_cluster_size=argv['min_cluster_size'],
            n_jobs=argv['n_jobs'], 
            cluster_features_version=argv['cluster_features_version'],
            classification_features_version=argv['classification_features_version'],
            stride=argv['stride']
        )
    else:
        X, y, metadata, tracks, labels = preprocess_dataset_for_training(
            path2dataset=path2dataset, 
            min_samples=argv['min_samples'], 
            max_eps=argv['max_eps'], 
            xi=argv['xi'], 
            min_cluster_size=argv['min_cluster_size'],
            n_jobs=argv['n_jobs'], 
            cluster_features_version=argv['cluster_features_version'],
            classification_features_version=argv['classification_features_version'],
        )

    tracks_filtered = [t for i, t in enumerate(tracks) if labels[i] > -1] 
    labels_filtered = [l for l in labels if l > -1]  

    cluster_centroids = None
    if argv['classification_features_version'] == 'v3' or argv['classification_features_version'] == 'v3_half':
        cluster_centroids = aoiextraction(tracks_filtered, labels_filtered)

    models = {
        'KNN' : KNeighborsClassifier,
        #'GP' : GaussianProcessClassifier,
        'SVM' : SVC,
    }
    
    parameters = {
        'KNN' : {'n_neighbors' : 15},
        #'GP' :  {},
        'SVM' : {'kernel' : 'rbf', 'probability' : True}
    }

    if not os.path.isdir(os.path.join(outdir, "tables")):
            os.mkdir(os.path.join(outdir, "tables"))

    if argv['batch_size'] is not None:
        batch_size = argv['batch_size']
        if X.shape[0] < batch_size:
            batch_size = X.shape[0]

    all_classes = np.array(list(set(y)))

    for clr in tqdm(models, desc="Classifier trained."):
        binaryModel = OneVSRestClassifierExtended(models[clr](**parameters[clr]), tracks, n_jobs=argv['n_jobs'])

        # if batch size is given, use partial_fit() method and train with minibatches
        if argv['batch_size'] is not None:
            try:
                iteration = 1
                for X_batch, y_batch in iter_minibatches(X, y, batch_size):
                    print(f"Iteration {iteration} started")
                    binaryModel.partial_fit(X_batch, y_batch, classes=all_classes, centroids=cluster_centroids)
                    iteration+=1
                print(f"\nTraining with batchsize: {batch_size:10d}.\n")
            except:
                print(f"\nClassifier {clr} does not have partial_fit() method, cant train with minibatches.")
                print(f"Training without minibatches. Batchsize is: {X.shape[0]:10d}\n")
                binaryModel.fit(X, y, centroids=cluster_centroids)            
        else:
            print(f"\nTraining without minibatches. Batchsize is: {X.shape[0]:10d}\n")
            binaryModel.fit(X, y, centroids=cluster_centroids)            

        # save models with names corresponding to the feature version and parameters
        if argv['classification_features_version'] == 'v1':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])), binaryModel)
        elif argv['classification_features_version'] == 'v1_half':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_from_half"), binaryModel) 
        elif argv['classification_features_version'] == 'v2':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v2"), binaryModel)
        elif argv['classification_features_version'] == 'v2_half':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v2_from_half"), binaryModel)
        elif argv['classification_features_version'] == 'v3':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v3"), binaryModel)
        elif argv['classification_features_version'] == 'v3_half':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v3_from_half"), binaryModel)
        elif argv['classification_features_version'] == 'v4':
            save_model(outdir, str("binary_"+clr+strfy_dict_params(parameters[clr])+"_v4"), binaryModel)
    
def BinaryClassificationTrain(classifier: str, path2db: str, **argv):
    """Deprecated, dont use.

    Will update in time.

    Args:
        classifier (str): _description_
        path2db (str): _description_
    """
    print("Warning: deprecated function, dont use.")
    print("Exiting...")
    exit(1)
    from classifier import BinaryClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    X_train, y_train, X_valid, y_valid, tracks = data_preprocessing_for_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'])
    table = pd.DataFrame()
    binaryModel = BinaryClassifier(X_train, y_train, tracks)
    if classifier == 'KNN':
        binaryModel.init_models(KNeighborsClassifier, n_neighbors=15)
    if classifier == 'MLP':
        binaryModel.init_models(MLPClassifier, max_iter=1000, solver="sgd")
    if classifier == 'SGD':
        binaryModel.init_models(SGDClassifier, loss="modified_huber")
    if classifier == 'GP':
        binaryModel.init_models(GaussianProcessClassifier)
    if classifier == 'GNB':
        binaryModel.init_models(GaussianNB)
    if classifier == 'SVM':
        binaryModel.init_models(SVC, kernel='rbf', probability=True)
    binaryModel.fit()
    accuracy_vector = binaryModel.validate(X_valid, y_valid, 0.8)
    table[classifier] = accuracy_vector # add col to pandas dataframe
    save_model(path2db, str("binary_"+classifier), binaryModel) 
    print(table.to_markdown()) # print out pandas dataframe in markdown table format.

def BinaryDecisionTreeClassification(path2dataset: str, min_samples: int, max_eps: float, xi: float, min_cluster_size: int, n_jobs: int, from_half=False):
    from classifier import BinaryClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import tree

    X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid = [], [], [], [] , [], [] 

    trackData = []

    threshold = 0.5
    
    if path2dataset.split(".")[-1] == "db":
        X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid, trackData = data_preprocessing_for_classifier(
            path2dataset, min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size, from_half=from_half)
    elif path2dataset.split(".")[-1] == "joblib":
        model = load_model(path2dataset)
        X_train, y_train, metadata_train, X_valid, y_valid, metadata_valid = data_preprocessing_for_classifier_from_joblib_model(
            model=model, min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size, n_jobs=n_jobs, from_half=from_half)
        trackData = model.trackData

    for d in range(2, 11):
        table_one = pd.DataFrame()
        # Initialize BinaryClassifier
        binaryModel = BinaryClassifier(X_train, y_train, trackData)
        binaryModel.init_models(DecisionTreeClassifier, max_depth=d)
        binaryModel.fit()
        # Validate BinaryClassifier
        predict_proba_balanced_accuracy = binaryModel.validate(X_valid, y_valid, threshold=threshold) # Validating the predict_proba() mathod, that returns back probability for every class
        predict_accuracy = binaryModel.validate_predictions(X_valid, y_valid, threshold=threshold) # validating predict() method, that returns only the highest predicted class
        # Create tables for accurcy
        table_one[f"Depth {d}"] = predict_proba_balanced_accuracy
        table_one.loc[0, f"Depth {d} multiclass average"] = np.average(predict_proba_balanced_accuracy)
        table_one.loc[0, f"Depth {d} one class prediction"] = predict_accuracy 
        # print out table in markdown
        print(f"Decision Tree depth {d} accuracy")
        print(table_one.to_markdown())

def validate_models(path2models: str, **argv):
    """Validate trained classifiers.

    Args:
        path2models (str): Path to parent directory containing models.
    """
    import datetime
    filenames = os.listdir(path2models)
    models = []
    classifier_names = []
    for n in filenames:
        if n.startswith("binary") and n.endswith(".joblib"):
            models.append(load_model(os.path.join(path2models, n)))
            classifier_names.append(n.split("_")[1].split(".")[0])

    table = pd.DataFrame()
    table2 = pd.DataFrame()
    probability_over_time = pd.DataFrame()

    
    if not os.path.isdir(os.path.join(*path2models.split("/")[:-1], "tables")):
        os.mkdir(os.path.join(*path2models.split("/")[:-1], "tables"))
    savepath = os.path.join(os.path.join(*path2models.split("/")[:-1], "tables"))

    _, _, _, X_valid, y_valid, metadata_valid = data_preprocessing_for_classifier_from_joblib_model(
        models[1], min_samples=argv["min_samples"], max_eps=argv["max_eps"], xi=argv["xi"],
        min_cluster_size=argv["min_cluster_size"], n_jobs=argv["n_jobs"],
        features_v2=argv['features_v2'], features_v2_half=argv['features_v2_half'])

    for clr, m in zip(classifier_names, models):
        top_picks = []
        for i in range(1,4):
            top_picks.append(m.validate_predictions(X_valid, y_valid, top=i))
        balanced_threshold = m.validate(X_valid, y_valid, argv['threshold'])
        # print(np.asarray(top_picks) )
        table[clr] = np.asarray(top_picks)
        table2[clr] = balanced_threshold

        probabilities = m.predict_proba(X_valid)
        for i in range(probabilities.shape[1]):
            probability_over_time[f"Class {i}"] = probabilities[:, i]
        probability_over_time["Time_Enter"] = metadata_valid[:, 0]
        probability_over_time["Time_Mid"] = metadata_valid[:, 1]
        probability_over_time["Time_Exit"] = metadata_valid[:, 2]
        probability_over_time["History_Length"] = metadata_valid[:, 3]
        probability_over_time["TrackID"] = metadata_valid[:, 4]
        probability_over_time["True_Class"] = y_valid  

        filename = os.path.join(savepath, f"{datetime.date.today()}_{clr}.xlsx")
        with pd.ExcelWriter(filename) as writer:
            probability_over_time.to_excel(writer, sheet_name="Probability_over_time")

    print("Top picks")
    print(table.to_markdown())
    print("Threshold")
    print(table2.to_markdown())
    print(table2.aggregate(np.average).to_markdown())

def true_class_under_threshold(predictions: np.ndarray, true_classes: np.ndarray, X: np.ndarray, threshold: float) -> np.ndarray:
    """Return numpy array of featurevectors that's predictions for their true class is under given threshold.

    Args:
        predictions (np.ndarray): Probability vectors. 
        true_classes (np.ndarray): Numpy array of the true classes ordered to feature vectors.
        X (np.ndarray): Feature vectors. 
        threshold (float): Threshold.

    Returns:
        np.ndarray: numpy array of feature vectors, that's true class's prediction probability is under threshold.
    """
    return_vector = []
    for i, pred in enumerate(predictions):
        if pred[true_classes[i]] < threshold:
            return_vector.append(X[i])
    return np.array(return_vector)

def all_class_under_threshold(predictions: np.ndarray, true_classes: np.ndarray, X: np.ndarray, threshold: float) -> np.ndarray:
    """Return numpy array of features that's predictions for all classes are under the given threshold.

    Args:
        predictions (np.ndarray): Probability vectors. 
        true_classes (np.ndarray): Numpy array of the true classes ordered to feature vectors.
        X (np.ndarray): Feature vectors. 
        threshold (float): Threshold.

    Returns:
        np.ndarray: numpy array of feature vectors, that's classes prediction probability is under threshold.
    """
    return_vector = []
    for i, preds in enumerate(predictions):
        renitent = True
        for pred in preds:
            if pred > threshold:
                renitent = False 
        if renitent:
            return_vector.append(X[i])
    return np.array(return_vector)

def investigateRenitent(path2model: str, threshold: float, **argv):
    """Filter out renitent predictions, that cant predict which class the detections is really in.

    Args:
        path2model (str): Path to model. 
    """
    model = load_model(path2model)
    _, _, _, X_test, y_test, _ = data_preprocessing_for_classifier_from_joblib_model(
        model, min_samples=argv["min_samples"], max_eps=argv["max_eps"], xi=argv["xi"],
        min_cluster_size=argv["min_cluster_size"], n_jobs=argv["n_jobs"])

    probas = model.predict_proba(X_test)

    renitent_vector = true_class_under_threshold(probas, y_test, X_test, threshold)

    renitent_vector_2 = all_class_under_threshold(probas, y_test, X_test, threshold)

    fig, ax = plt.subplots(1, 2)

    if len(renitent_vector) > 0:
        ax[0].set_title(f"Renitent: true class under threshold {threshold}: {len(renitent_vector)}")
        ax[0].scatter(renitent_vector[:, 0], 1 - renitent_vector[:, 1], s=2.5, c='g')
        ax[0].scatter(renitent_vector[:, 4], 1 - renitent_vector[:, 5], s=2.5)
        ax[0].scatter(renitent_vector[:, 6], 1 - renitent_vector[:, 7], s=2.5, c='r')
        print(f"There are {len(renitent_vector)} renitent detections out of {len(X_test)}.")
    else:
        print(f"Renitent: true class under threshold {threshold}")

    if len(renitent_vector_2) > 0:
        ax[1].set_title(f"Renitent: classes under threshold {threshold}: {len(renitent_vector_2)}")
        ax[1].scatter(renitent_vector_2[:, 0], 1 - renitent_vector_2[:, 1], s=2.5, c='g')
        ax[1].scatter(renitent_vector_2[:, 4], 1 - renitent_vector_2[:, 5], s=2.5)
        ax[1].scatter(renitent_vector_2[:, 6], 1 - renitent_vector_2[:, 7], s=2.5, c='r')
        print(f"There are {len(renitent_vector_2)} renitent detections out of {len(X_test)}.")
    else:
        print(f"Renitent: classes under threshold {threshold}")
        
    plt.show()

def plot_decision_tree(path2model: str):
    """Draw out the decision tree in a tree graph.

    Args:
        path2model (str): Path to the joblib binary model file.
    """
    from sklearn.tree import plot_tree
    model = load_model(path2model=path2model)
    for i, m in enumerate(model.models_):
        print(f"Class {i}")
        plot_tree(m)
        plt.show()

def cross_validate(path2dataset: str, outputPath: str = None, train_ratio=0.75, seed=1, n_splits=5, n_jobs=18, estimator_params_set=1, classification_features_version: str = "v1", stride: int = 15):
    """Calculate classification model accuracy with cross validation method.

    Args:
        path2dataset (str): Path to dataset. File with joblib extension. 
        train_ratio (float, optional): Split ratio of the tracks. Defaults to 0.75.
        seed (int, optional): Seed for reproducable track splitting. Defaults to 1.
        n_splits (int, optional): Number of splits to perform with k-fold cross validation method. Defaults to 5.
        n_jobs (int, optional): Number of jobs to run. Defaults to 18.
    """
    from processing_utils import (
                                    load_joblib_tracks, 
                                    random_split_tracks, 
                                    make_feature_vectors_version_two, 
                                    make_feature_vectors_version_two_half, 
                                    make_features_for_classification_velocity_time, 
                                    make_features_for_classification_velocity_time_second_half,
                                    make_feature_vectors_version_three, 
                                    make_feature_vectors_version_three_half,
                                    make_feature_vectors_version_four
                                )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import SGDClassifier, SGDOneClassSVM
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC
    from classifier import OneVSRestClassifierExtended
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import cross_val_score, cross_validate
    from sklearn.metrics import top_k_accuracy_score, make_scorer, balanced_accuracy_score
    from visualizer import aoiextraction

    # load tracks from joblib file
    # tracks stored as list[dict]
    # {
    #   "track": t,
    #   "class": c
    # }
    tracks = load_joblib_tracks(path2dataset)
    tracks_filteted = []
    for t in tracks:
        if t["class"] != -1:
            tracks_filteted.append(t)

    # shuffle tracks, and separate into a train and test dataset
    train, test = random_split_tracks(tracks_filteted, train_ratio, seed)

    tracks_train = [t["track"] for t in train]
    labels_train = np.array([t["class"] for t in train])
    tracks_test = [t["track"] for t in test]
    labels_test = np.array([t["class"] for t in test])

    cluster_centroids = None
    fit_params = None

    if classification_features_version == "v1":
        X_train, y_train, metadata_train = make_features_for_classification_velocity_time(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_features_for_classification_velocity_time(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v1_half":
        X_train, y_train, metadata_train = make_features_for_classification_velocity_time_second_half(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_features_for_classification_velocity_time_second_half(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v2":
        X_train, y_train, metadata_train = make_feature_vectors_version_two(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_two(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v2_half":
        X_train, y_train, metadata_train = make_feature_vectors_version_two_half(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_two_half(trackedObjects=tracks_test, k=6, labels=labels_test)
    elif classification_features_version == "v3":
        X_train, y_train, metadata_train = make_feature_vectors_version_three(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_three(trackedObjects=tracks_test, k=6, labels=labels_test)
        cluster_centroids = aoiextraction([t["track"] for t in tracks_filteted], [t["class"] for t in tracks_filteted]) 
        fit_params = {
            'centroids' : cluster_centroids
        }
    elif classification_features_version == "v3_half":
        X_train, y_train, metadata_train = make_feature_vectors_version_three_half(trackedObjects=tracks_train, k=6, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_three_half(trackedObjects=tracks_test, k=6, labels=labels_test)
        cluster_centroids = aoiextraction([t["track"] for t in tracks_filteted], [t["class"] for t in tracks_filteted]) 
        fit_params = {
            'centroids' : cluster_centroids
        }
    elif classification_features_version == "v4":
        X_train, y_train, metadata_train = make_feature_vectors_version_four(trackedObjects=tracks_train, max_stride=stride, labels=labels_train)
        X_test, y_test, metadata_train = make_feature_vectors_version_four(trackedObjects=tracks_test, max_stride=stride, labels=labels_test)

    models = {
        'KNN' : KNeighborsClassifier,
        #'GP' : GaussianProcessClassifier,
        #'GNB' : GaussianNB,
        #'MLP' : MLPClassifier,
        #'SGD_modified_huber' : SGDClassifier,
        #'SGD_log_loss' : SGDClassifier,
        'SVM' : SVC,
        #'DT' : DecisionTreeClassifier
    }
    
    parameters = [{
                    'KNN' : {'n_neighbors' : 15},
                    #'GP' :  {},
                    #'GNB' : {},
                    #'MLP' : {'max_iter' : 1000, 'solver' : 'sgd'},
                    #'SGD_modified_huber' : {'loss' : 'modified_huber'},
                    #'SGD_log_loss' : {'loss' : 'log_loss'},
                    'SVM' : {'kernel' : 'rbf', 'probability' : True},
                    #'DT' : {} 
                }, {
                    'KNN' : {'n_neighbors' : 3},
                    #'GP' :  {},
                    #'GNB' : {},
                    #'MLP' : {'max_iter' : 2000, 'solver' : 'sgd'},
                    #'SGD_modified_huber' : {'loss' : 'modified_huber', 'max_iter' : 2000},
                    #'SGD_log_loss' : {'loss' : 'log_loss', 'max_iter' : 2000},
                    'SVM' : {'kernel' : 'linear', 'probability' : True},
                    #'DT' : {} 
                }, {
                    'KNN' : {'n_neighbors' : 1},
                    #'GP' :  {},
                    #'GNB' : {},
                    #'MLP' : {'max_iter' : 3000, 'solver' : 'sgd'},
                    #'SGD_modified_huber' : {'loss' : 'modified_huber', 'max_iter' : 3000},
                    #'SGD_log_loss' : {'loss' : 'log_loss', 'max_iter' : 3000},
                    'SVM' : {'kernel' : 'linear', 'probability' : True},
                    #'DT' : {} 
                }, {
                    'KNN' : {'n_neighbors' : 7},
                    #'GP' :  {},
                    #'GNB' : {},
                    #'MLP' : {'max_iter' : 4000, 'solver' : 'sgd'},
                    #'SGD_modified_huber' : {'loss' : 'modified_huber', 'max_iter' : 16000},
                    #'SGD_log_loss' : {'loss' : 'log_loss', 'max_iter' : 16000},
                    'SVM' : {'kernel' : 'rbf', 'probability' : True},
                    #'DT' : {} 
                }]
    
    splits = np.append(np.arange(1,6,1), ["Max split", "Mean", "Standart deviation"])
    basic_table = pd.DataFrame()
    balanced_table = pd.DataFrame()
    top_1_table = pd.DataFrame()
    top_2_table = pd.DataFrame()
    top_3_table = pd.DataFrame()
    final_test_basic = pd.DataFrame()
    final_test_balanced = pd.DataFrame()
    final_test_top_k_idx = ["Top_1", "Top_2", "Top_3"]
    final_test_top_k = pd.DataFrame()

    basic_table["Split"] = splits
    balanced_table["Split"] = splits
    top_1_table["Split"] = splits
    top_2_table["Split"] = splits
    top_3_table["Split"] = splits
    final_test_top_k["Top"] = final_test_top_k_idx

    parameters_table = pd.DataFrame(parameters[estimator_params_set-1])

    # makeing top_k scorer callables, to be able to set their k parameter
    #top_1_scorer = make_scorer(top_k_accuracy_score, k=1)
    #top_2_scorer = make_scorer(top_k_accuracy_score, k=2)
    #top_3_scorer = make_scorer(top_k_accuracy_score, k=3)
    top_k_scorers = {
        'top_1' : make_scorer(top_k_accuracy_score, k=1, needs_proba=True),
        'top_2' : make_scorer(top_k_accuracy_score, k=2, needs_proba=True),
        'top_3' : make_scorer(top_k_accuracy_score, k=3, needs_proba=True) 
    }

    print(f"\nTraining dataset size: {X_train.shape[0]}")
    print(f"Validation dataset size: {X_test.shape[0]}\n")

    t1 = time.time()
    for m in tqdm(models, desc="Cross validate models"):
        clf = OneVSRestClassifierExtended(estimator=models[m](**parameters[estimator_params_set-1][m]), tracks=tracks_train, n_jobs=n_jobs)

        basic_scores = cross_val_score(clf, X_train, y_train, cv=n_splits, fit_params=fit_params, n_jobs=n_jobs)
        basic_table[m] = np.append(basic_scores, [np.max(basic_scores), basic_scores.mean(), basic_scores.std()]) 

        balanced_scores = cross_val_score(clf, X_train, y_train, cv=n_splits, scoring='balanced_accuracy', fit_params=fit_params, n_jobs=n_jobs)
        balanced_table[m] = np.append(balanced_scores, [np.max(balanced_scores), balanced_scores.mean(), balanced_scores.std()]) 

        top_k_scores = cross_validate(clf, X_train, y_train, scoring=top_k_scorers, cv=5, fit_params=fit_params, n_jobs=n_jobs)
        top_1_table[m] = np.append(top_k_scores['test_top_1'], [np.max(top_k_scores['test_top_1']), top_k_scores['test_top_1'].mean(), top_k_scores['test_top_1'].std()])
        top_2_table[m] = np.append(top_k_scores['test_top_2'], [np.max(top_k_scores['test_top_2']), top_k_scores['test_top_2'].mean(), top_k_scores['test_top_2'].std()])
        top_3_table[m] = np.append(top_k_scores['test_top_3'], [np.max(top_k_scores['test_top_3']), top_k_scores['test_top_3'].mean(), top_k_scores['test_top_3'].std()])

        clf.fit(X_train, y_train, centroids=cluster_centroids)

        y_pred = clf.predict(X_test)
        y_pred_2 = clf.predict_proba(X_test)

        #final_balanced = clf.validate(X_test, y_test, threshold=0.5, centroids=cluster_centroids)
        #final_balanced_avg = np.average(final_balanced)
        #final_balanced_std = np.std(final_balanced)
        #final_test_balanced["Class"] = np.append(np.arange(len(final_balanced)), ["Mean", "Standart deviation"])
        #final_test_balanced[m] = np.append(final_balanced, [final_balanced_avg, final_balanced_std])

        final_top_k = []
        for i in range(1,4):
            final_top_k.append(top_k_accuracy_score(y_test, y_pred_2, k=i, labels=list(set(y_train))))
        final_test_top_k[m] = final_top_k

        final_basic = np.array([clf.score(X_test, y_test)])
        final_test_basic[m] = final_basic

        final_balanced = balanced_accuracy_score(y_test, y_pred)
        final_test_balanced[m] = np.array([final_balanced])

    t2 = time.time()
    td = t2 - t1
    print("\n*Time: %d s*" % td)

    print("\n#### Classifier parameters\n")
    print(parameters[estimator_params_set-1])

    print("\n#### Cross-val Basic accuracy\n")
    print(basic_table.to_markdown())
    
    print("\n#### Cross-val Balanced accuracy\n")
    print(balanced_table.to_markdown())

    print("\n#### Cross-val Top 1 accuracy\n")
    print(top_1_table.to_markdown())

    print("\n#### Cross-val Top 2 accuracy\n")
    print(top_2_table.to_markdown())

    print("\n#### Cross-val Top 3 accuracy\n")
    print(top_3_table.to_markdown())

    print("\n#### Test set basic\n")
    print(final_test_basic.to_markdown())

    print("\n#### Test set balanced\n")
    print(final_test_balanced.to_markdown())

    print("\n#### Test set top k\n")
    print(final_test_top_k.to_markdown())

    if outputPath is not None:
        with pd.ExcelWriter(outputPath) as writer:
            parameters_table.to_excel(writer, sheet_name="Classifier parameters")
            basic_table.to_excel(writer, sheet_name="Cross Validation Basic scores")
            balanced_table.to_excel(writer, sheet_name="Cross Validation Balanced scores")
            top_1_table.to_excel(writer, sheet_name="Cross Validation Top 1 scores")
            top_2_table.to_excel(writer, sheet_name="Cross Validation Top 2 scores")
            top_3_table.to_excel(writer, sheet_name="Cross Validation Top 3 scores")
            final_test_basic.to_excel(writer, sheet_name="Validation set Basic scores")
            final_test_balanced.to_excel(writer, sheet_name="Validation set Balanced scores")
            final_test_top_k.to_excel(writer, sheet_name="Validation set Top K scores")

    print()
    return basic_table, balanced_table, top_1_table, top_2_table, top_3_table, final_test_basic, final_test_balanced, final_test_top_k
    
# submodule functions
def train_binary_classifiers_submodule(args):
    train_binary_classifiers(args.database, args.outdir, 
                            min_samples=args.min_samples, 
                            max_eps=args.max_eps,xi=args.xi, 
                            min_cluster_size=args.min_samples, n_jobs=args.n_jobs,
                            cluster_features_version=args.cluster_features_version,
                            classification_features_version=args.classification_features_version,
                            stride=args.stride,
                            batch_size=args.batchsize)

def cross_validation_submodule(args):
    cross_validate(args.database, args.output, 
                args.train_ratio, 
                args.seed, n_jobs=args.n_jobs, 
                estimator_params_set=args.param_set, 
                #cluster_features_version=args.cluster_features_version,
                classification_features_version=args.classification_features_version,
                stride=args.stride)

def investigate_renitent_features(args):
    investigateRenitent(args.model, args.threshold, 
                        min_samples=args.min_samples, 
                        max_eps=args.max_eps, xi=args.xi, 
                        min_cluster_size=args.min_samples, 
                        n_jobs=args.n_jobs)

def main():
    import argparse
    argparser = argparse.ArgumentParser("Train, validate and test for renitent detection.")
    argparser.add_argument("--n_jobs", type=int, help="Number of processes.", default=1)

    submodule_parser = argparser.add_subparsers(help="Program functionalities.")

    # add subcommands for training binary classifiers
    train_binary_classifiers_parser = submodule_parser.add_parser(
        "train",
        help="Run Classification on dataset, but not as a multi class classification, rather do "
             "binary classification for each cluster."
    )
    train_binary_classifiers_parser.add_argument("-db", "--database", help="Path to database file. This should be an unclustered joblib dataset file.", type=str)
    train_binary_classifiers_parser.add_argument("--outdir", "-o", help="Output directory path.", type=str)
    train_binary_classifiers_parser.add_argument("--min_samples", default=10, type=int, 
        help="OPTICS parameter: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.")
    train_binary_classifiers_parser.add_argument("--max_eps", type=float, default=0.2, 
        help="OPTICS parameter: The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    train_binary_classifiers_parser.add_argument("--xi", type=float, default=0.15, 
        help="OPTICS parameter: Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.")
    train_binary_classifiers_parser.add_argument("--min_cluster_size", default=10, type=float,
        help="OPTICS parameter: Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).")
    train_binary_classifiers_parser.add_argument("--cluster_features_version", choices=["4D", "6D"], help="Choose which version of features to use for clustering.", default="6D")
    train_binary_classifiers_parser.add_argument("--classification_features_version", choices=["v1", "v1_half", "v2", "v2_half", "v3", "v3_half", "v4"], help="Choose which version of features to use for classification.", default="v1")
    train_binary_classifiers_parser.add_argument("--stride", default=15, type=int, help="Set stride value of classification features v4.")
    train_binary_classifiers_parser.add_argument("--batchsize", type=int, default=None, help="Set training batch size.")
    train_binary_classifiers_parser.set_defaults(func=train_binary_classifiers_submodule)

    # add subcommands for cross validating classifiers 
    cross_validation_parser = submodule_parser.add_parser(
        "cross-validation",
        help="Run cross validation with given dataset."
    )
    cross_validation_parser.add_argument("-db", "--database", help="Path to database file. This should be an already clustered joblib dataset file.", type=str)
    cross_validation_parser.add_argument("--output", "-o", help="Output file path, make sure that the directory of the outputted file exists.", type=str)
    cross_validation_parser.add_argument("--train_ratio", help="Size of the train dataset. (0-1 float)", type=float, default=0.75)
    cross_validation_parser.add_argument("--seed", help="Seed for random number generator to be able to reproduce dataset shuffle.", type=int, default=1)
    cross_validation_parser.add_argument("--param_set", help="Choose between the parameter sets that will be given to the classifiers.", type=int, choices=[1,2,3,4], default=1)
    cross_validation_parser.add_argument("--classification_features_version", choices=["v1", "v1_half", "v2", "v2_half", "v3", "v3_half", "v4"], help="Choose which version of features to use for classification.", default="v1")
    cross_validation_parser.add_argument("--stride", default=15, type=int, help="Set stride value of classification features v4.")
    #cross_validation_parser.add_argument("--cluster_features_version", choices=["4D", "6D"], help="Choose which version of features to use for clustering.", default="6D")
    cross_validation_parser.set_defaults(func=cross_validation_submodule)

    # add subcommands for renitent investigation module
    renitent_filter_parser = submodule_parser.add_parser(
        "renitent-filter",
        help="Look at detections, that cant be predicted above a given threshold value."
    )
    renitent_filter_parser.add_argument("--model", help="Trained classifier.", type=str)
    renitent_filter_parser.add_argument("--threshold", type=float, default=0.5, help="Balanced accuracy threshold.")
    renitent_filter_parser.add_argument("--min_samples", default=10, type=int, 
        help="OPTICS parameter: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.")
    renitent_filter_parser.add_argument("--max_eps", type=float, default=0.2,
        help="OPTICS parameter: The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    renitent_filter_parser.add_argument("--xi", type=float, default=0.15,
        help="OPTICS parameter: Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.")
    renitent_filter_parser.add_argument("--min_cluster_size", default=10, type=float, 
        help="OPTICS parameter: Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction" 
             "of the number of samples (rounded to be at least 2).")
    renitent_filter_parser.set_defaults(func=investigate_renitent_features)

    args = argparser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
