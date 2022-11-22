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
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from utils import load_model, save_model, data_preprocessing_for_calibrated_classifier, data_preprocessing_for_classifier, data_preprocessing_for_classifier_from_joblib_model, checkDir

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
                    fp +=1
    #TODO
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
    """Run all of the classification methods implemented.

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
    from classifier import BinaryClassifier
    from sklearn.tree import DecisionTreeClassifier

    X_train, y_train, time_train, X_valid, y_valid, time_test, tracks = [], [], [], [], [], [], []

    if path2model is not None:
        model = load_model(path2model)
        tracks = model.trackData
        X_train, y_train, time_train, X_valid, y_valid, time_test = data_preprocessing_for_classifier_from_joblib_model(model, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'], from_half=argv['from_half'])
    else:
        X_train, y_train, time_train, X_valid, y_valid, time_test, tracks = data_preprocessing_for_classifier(path2db, min_samples=argv['min_samples'], 
                                                            max_eps=argv['max_eps'], 
                                                            xi=argv['xi'], 
                                                            min_cluster_size=argv['min_cluster_size'],
                                                            n_jobs=argv['n_jobs'], from_half=argv['from_half'])

    models = {
        'KNN' : KNeighborsClassifier,
        'GP' : GaussianProcessClassifier,
        'GNB' : GaussianNB,
        'MLP' : MLPClassifier,
        'SGD' : SGDClassifier,
        'SVM' : SVC,
        'DT' : DecisionTreeClassifier
    }

    table = pd.DataFrame()
    table2 = pd.DataFrame()
    probability_over_time = pd.DataFrame()

    if not os.path.isdir(os.path.join('research_data', path2db.split('/')[-1].split('.')[0], "tables")):
            os.mkdir(os.path.join('research_data', path2db.split('/')[-1].split('.')[0], "tables"))
    savepath = os.path.join(os.path.join('research_data', path2db.split('/')[-1].split('.')[0], "tables"))

    for clr in models:
        binaryModel = BinaryClassifier(X_train, y_train, tracks)
        if clr == 'KNN':
            binaryModel.init_models(models[clr], n_neighbors=15)
        elif clr == 'MLP':
            binaryModel.init_models(models[clr], max_iter=1000, solver="sgd")
        elif clr == 'SGD':
            binaryModel.init_models(models[clr], loss="modified_huber")
        elif clr == 'SVM':
            binaryModel.init_models(models[clr], kernel='rbf', probability=True)
        else:
            binaryModel.init_models(models[clr])
        binaryModel.fit()

        balanced_toppicks = binaryModel.validate_predictions(X_valid, y_valid, argv['threshold'])
        balanced_threshold = binaryModel.validate(X_valid, y_valid, argv['threshold'])

        table.loc[0, clr] = balanced_toppicks 
        table2[clr] = balanced_threshold

        probabilities = binaryModel.predict_proba(X_valid)
        for i in range(probabilities.shape[1]):
            probability_over_time[f"Class {i}"] = probabilities[:, i]
        probability_over_time["Time_Enter"] = time_test[:, 0]
        probability_over_time["Time_Mid"] = time_test[:, 1]
        probability_over_time["Time_Exit"] = time_test[:, 2]
        filename = os.path.join(savepath, f"{clr}.xlsx")
        with pd.ExcelWriter(filename) as writer:
            probability_over_time.to_excel(writer, sheet_name="Probability_over_time")
        save_model(path2db, str("binary_"+clr), binaryModel) 

    print("Top picks")
    print(table.to_markdown())
    print("Threshold")
    print(table2.to_markdown())
    print(table2.aggregate(np.average).to_markdown())
    
def BinaryClassificationTrain(classifier: str, path2db: str, **argv):
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

    X_train, y_train, time_train, X_valid, y_valid, time_valid = [], [], [], [] , [], [] 

    trackData = []

    threshold = 0.5
    
    if path2dataset.split(".")[-1] == "db":
        X_train, y_train, time_train, X_valid, y_valid, time_valid, trackData = data_preprocessing_for_classifier(path2dataset, min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size)
    elif path2dataset.split(".")[-1] == "joblib":
        model = load_model(path2dataset)
        X_train, y_train, time_train, X_valid, y_valid, time_valid = data_preprocessing_for_classifier_from_joblib_model(model=model, min_samples=min_samples, max_eps=max_eps, xi=xi, min_cluster_size=min_cluster_size, n_jobs=n_jobs, from_half=from_half)
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

    _, _, _, X_valid, y_valid, time_valid = data_preprocessing_for_classifier_from_joblib_model(models[0], min_samples=argv["min_samples"], max_eps=argv["max_eps"], xi=argv["xi"], min_cluster_size=argv["min_cluster_size"], n_jobs=argv["n_jobs"])

    for clr, m in zip(classifier_names, models):
        balanced_toppicks = m.validate_predictions(X_valid, y_valid, argv['threshold'])
        balanced_threshold = m.validate(X_valid, y_valid, argv['threshold'])

        table.loc[0, clr] = balanced_toppicks 
        table2[clr] = balanced_threshold

        probabilities = m.predict_proba(X_valid)
        for i in range(probabilities.shape[1]):
            probability_over_time[f"Class {i}"] = probabilities[:, i]
        probability_over_time["Time_Enter"] = time_valid[:, 0]
        probability_over_time["Time_Mid"] = time_valid[:, 1]
        probability_over_time["Time_Exit"] = time_valid[:, 2]

        filename = os.path.join(savepath, f"{datetime.date.today()}_{clr}.xlsx")
        with pd.ExcelWriter(filename) as writer:
            probability_over_time.to_excel(writer, sheet_name="Probability_over_time")

    print("Top picks")
    print(table.to_markdown())
    print("Threshold")
    print(table2.to_markdown())
    print(table2.aggregate(np.average).to_markdown())

def investigateRenitent(path2model: str):
    """Filter out renitent predictions, that cant predict which class the detections is really in.

    Args:
        path2model (str): Path to model. 
    """
    start = time.time()
    model = load_model(path2model)
    _, _, _, X_test, y_test, time_test = data_preprocessing_for_classifier_from_joblib_model(model)
    probas = model.predict_proba(X_test)

    sure_vector = []
    renitent_vector = []

    for i, proba_vector in enumerate(probas):
        unsure_counter = 0
        max = np.max(proba_vector)
        for j, p in enumerate(proba_vector):
            if p <= ( max + max * 0.1 ) and p >= ( max - max * 0.1):
                unsure_counter += 1
        if unsure_counter >= 2:
            renitent_vector.append(X_test[i])
        else:
            sure_vector.append(X_test[i])
    print(f"Processing time: {time.time()-start}")
  
    renitent_vector = np.array(renitent_vector) 
    sure_vector = np.array(sure_vector)
  
    fig, ax = plt.subplots(1,2, figsize=(20,10))
    
    if len(renitent_vector) > 0:
        ax[0].scatter(renitent_vector[:, 0], 1 - renitent_vector[:, 1], s=2.5, c='g')
        ax[0].scatter(renitent_vector[:, 4], 1 - renitent_vector[:, 5], s=2.5)
        ax[0].scatter(renitent_vector[:, 6], 1 - renitent_vector[:, 7], s=2.5, c='r')
        ax[0].set_title(f"Renitent predictions {len(renitent_vector)}")

    ax[1].scatter(sure_vector[:, 0], 1 - sure_vector[:, 1], s=2.5, c='g')
    ax[1].scatter(sure_vector[:, 4], 1 - sure_vector[:, 5], s=2.5)
    ax[1].scatter(sure_vector[:, 6], 1 - sure_vector[:, 7], s=2.5, c='r')
    ax[1].set_title(f"Sure predictions {len(probas)-len(renitent_vector)}")
  
    plt.show()
  
    print(f"Solid predictions: {len(probas)-len(renitent_vector)}")
    print(f"Unsure predictions: {len(renitent_vector)}")

def main():
    import argparse
    from classification import Classification, ClassificationWorker, CalibratedClassification, CalibratedClassificationWorker, BinaryClassificationTrain, BinaryClassificationWorkerTrain, validate_models, investigateRenitent
    argparser = argparse.ArgumentParser("Analyze results of main program. Make and save plots. Create heatmap or use clustering on data stored in the database.")
    argparser.add_argument("-db", "--database", help="Path to database file.")
    argparser.add_argument("--threshold", type=float, default=0.5, help="Threshold value for filtering algorithm that filters out the best detections.")
    argparser.add_argument("--n_jobs", type=int, help="Number of processes.", default=1)
    argparser.add_argument("--eps", default=0.1, type=float, help="DBSCAN and OPTICS_DBSCAN parameter: The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    argparser.add_argument("--min_samples", default=10, type=int, help="DBSCAN and OPTICS parameter: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.")
    argparser.add_argument("--max_eps", help="OPTICS parameter: The maximum distance between two samples for one to be considered as in the neighborhood of the other.", type=float, default=0.2)
    argparser.add_argument("--xi", help="OPTICS parameter: Determines the minimum steepness on the reachability plot that constitutes a cluster boundary.", type=float, default=0.15)
    argparser.add_argument("--min_cluster_size", default=10, type=float, help="OPTICS parameter: Minimum number of samples in an OPTICS cluster, expressed as an absolute number or a fraction of the number of samples (rounded to be at least 2).")
    argparser.add_argument("--Classification", help="Train model with classification.", default=False, choices=['KNN', 'SGD', 'GP', 'GNB', 'MLP', 'VOTE', 'SVM', 'DT'])
    argparser.add_argument("--CalibratedClassification", help="Train model with calibrated classification.", default=False, choices=['KNN', 'SGD', 'GP', 'GNB', 'MLP', 'VOTE'])
    argparser.add_argument("--n_neighbours", help="KNN parameter: Number of neighbours for clustering.", type=int)
    argparser.add_argument("--ClassificationWorker", help="Runs all avaliable Classifications and Validate them.", default=False, action="store_true")
    argparser.add_argument("--CalibratedClassificationWorker", help="Runs all avaliable Classifications calibrated and Validate them.", default=False, action="store_true")
    argparser.add_argument("--BinaryClassificationWorkerTrain", default=False, action="store_true", help="Run Classification on dataset, but not as a multi class classification, rather do binary classification for each cluster.")
    argparser.add_argument("--BinaryClassificationTrain", help="Train model with binary classification.", default=False, choices=['KNN', 'SGD', 'GP', 'GNB', 'MLP', 'SVM'])
    argparser.add_argument("--from_half", help="Use thid flag, if want to make feature vectors only from second half of trajectories history.", action="store_true", default=False)
    argparser.add_argument("--model", help="Load classifier.", type=str, default=None)
    argparser.add_argument("--validate_classifiers", help="Validate accuracy of trained classifier models.", action="store_true", default=False)
    argparser.add_argument("--plot_renitent_features", help="Draw diagram of renitent feature vectors.", action="store_true", default=False)
    argparser.add_argument("--decision_tree_accuracy_over_depth", action="store_true", default=False)
    args = argparser.parse_args()
    if args.database is not None:
        checkDir(args.database)
    if args.Classification:
        Classification(args.Classification, args.database, min_samples=args.min_samples, max_eps=args.max_eps, xi=args.xi, min_cluster_size=args.min_samples, n_jobs=args.n_jobs)
    if args.ClassificationWorker:
        ClassificationWorker(args.database, min_samples=args.min_samples, max_eps=args.max_eps, xi=args.xi, min_cluster_size=args.min_samples, n_jobs=args.n_jobs)
    if args.CalibratedClassification:
        CalibratedClassification(args.CalibratedClassification, args.database, min_samples=args.min_samples, max_eps=args.max_eps, xi=args.xi, min_cluster_size=args.min_samples, n_jobs=args.n_jobs)
    if args.CalibratedClassificationWorker:
        CalibratedClassificationWorker(args.database, min_samples=args.min_samples, max_eps=args.max_eps, xi=args.xi, min_cluster_size=args.min_samples, n_jobs=args.n_jobs)
    if args.BinaryClassificationWorkerTrain:
        BinaryClassificationWorkerTrain(args.database, args.model, min_samples=args.min_samples, max_eps=args.max_eps, xi=args.xi, min_cluster_size=args.min_samples, n_jobs=args.n_jobs, threshold=args.threshold, from_half=args.from_half)
    if args.BinaryClassificationTrain:
        BinaryClassificationTrain(args.BinaryClassification, args.database, min_samples=args.min_samples, max_eps=args.max_eps, xi=args.xi, min_cluster_size=args.min_samples, n_jobs=args.n_jobs)
    if args.plot_renitent_features:
        investigateRenitent(args.model)
    if args.validate_classifiers and args.threshold:
        validate_models(args.model, min_samples=args.min_samples, max_eps=args.max_eps, xi=args.xi, min_cluster_size=args.min_samples, n_jobs=args.n_jobs, threshold=args.threshold)
    if args.decision_tree_accuracy_over_depth:
        if args.database:
            BinaryDecisionTreeClassification(args.database, args.min_samples, args.max_eps, args.xi, args.min_cluster_size, args.n_jobs, args.from_half)
        elif args.model:
            BinaryDecisionTreeClassification(args.model, args.min_samples, args.max_eps, args.xi, args.min_cluster_size, args.n_jobs, args.from_half)

if __name__ == "__main__":
    main()