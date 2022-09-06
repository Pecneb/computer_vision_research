# computer_vision_research

Predicting trajectories of objects

## Abstract

**TODO**: Abstract

**Notice:** linear regression implemented, very primitive, but working  

## Darknet

For detection, I used darknet neural net and YOLOV4 pretrained model. [[1]](#1)
In order to be able to use the darknet api, build from source with the LIB flag on. Then copy libdarknet.so to root dir of the project. (My Makefile to build darknet can be found in the darknet_config_files directory)  

**Notice:** Using the yolov4-csp-x-swish.cfg and weights with RTX 3070 TI is doing 26 FPS with 69.9% precision, this is the most stable detection so far, good base for tracking and predicting  

For Darknet, I wrote an API hldnapi.py, that makes object detection more easier. cvimg2detections(img) it takes only an opencv img and returns the detections in format [label, confidence, xywh]

## YOLOV7

Yolov7 is the most recent version of YOLO. Darknet is no more, the source code of the neural net is in PyTorch. [Original-Repository](https://github.com/WongKinYiu/yolov7) [[2]](#2). To work with my framework, I read the whole codebase of Yolov7. I wrote yolov7api.py, function load_model(device, weights, imgsz, classify) can load the desired yolo model, if GPU is used half precision can be used (FP16 instead of FP32), detect(img) takes an opencv image as argument, it can take a lot more arguments, but those are only for parametization, there are default values set for those arguments, that are tested. The image has to be resized to the size of the NeuralNet. After the model is loaded, we can input the resized image to the neural net. The results are a matrix shaped (number of input images, number of detections, 6). A detection is a vector of [x, y, x, y, confidence, class] (first xy is top-left, second xy is bottom-right). The raw output of the neural net has to be resized to fit the original image. The output is still not good for my framework.
The output have to be converted to a matrix of shape(number of detections, 3) what is looks like [label, confidence, (xywh)] xywh is center xy coordinates and width, height of bbox.  

**NOTICE**: If pytorch throws this error: RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB (GPU 0; X.XX GiB total capacity; X.XX MiB already allocated; X.XX GiB free; X.XX GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF. Then set environment variable PYTORCH_CUDA_ALLOC_CONF to `PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"`, if this does not solve the problem, play with the `max_split_size_mb`, try to give it other sizes.

### Installation

Download yolov7 weights file from [yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt), then copy or move it to yolov7 directory.

Create conda environment and add yolov7 to PYTHONPATH.

```shell
conda create -n <insert name here> python=3.9
conda install pytorch torchvision torchaudio cudatoolkit=11.6 opencv matplotlib pandas tqdm pyyaml seaborn -c conda-forge -c pytorch
export PYTHONPATH="${PYTHONPATH}:<PATH to YOLOV7 directory>"
```

The setup of PYTHONPATH variable is very important, because python will throw a module error. To not have to set this environment variable every time use `conda env config vars set PYTHONPATH=${PYTHONPATH}:<PATH to YOLOV7 directory>"` command.  

In case this is not working, I implemented a gpu memory freeing function, which is called when yolov7 is imported or yolov7 model is loaded.  

## Tracking of detected objects

**Base idea**: track objects from one frame to the other, based on x and y center coordinates. This solution require very minimal resources.  

**Euclidean distances**: This should be more precise, but require a lot more computation. Have to examine this technique further to get better results.  

**Deep-SORT**: Simple Online and Realtime Tracking with convolutonal neural network. Pretty much based on Kalmanfilter. See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information. [[3]](#3)  

### Determining wheter an object moving or not

**Temporary solution**: Calculating the tracking history's last and the first detection's euclidean distance.  

```python
self.isMoving = ((self.history[0].X-self.history[-1].X)**2 + (self.history[0].Y-self.history[-1].Y)**2)**(1/2) > 7.0  
```

### Throw away old detections and trackings

This can save read, write time and memory.  

**HistoryDepth**: Implemented a historyDepth variable, that determines how long back in time should we track an objects detection data. With this, we can throw away old trackings if they are not on screen any more.  

**Bug**: If the main.py script is running for a long time, the number of tracks are piling up. This seems to slow down the program gradually.  
**Bug**: The above mentioned bug is in correlation with another problem, few tracks from the piled up tracks, are logged out to the terminal output, although only moving objects should be logged. The velocity and accelaration of these objects are stuck and not changing, sometimes these values are way high to be real.
**Fixes**: To filter out the bugged tracks, a bugged counter field is being added to the dataManagementClasses.TrackedObject() class. The counter is incremented, when the velocities are the same as the velocities from the earlier detection. Then the track is removed from the history, when the counter reaches the given maxAge value.

## Predicting trajectories of moving objects

### Linear Regression

Using **Scikit Learn Linear Models**  

```python
model = linear_model.RANSACRegressor(base_estimator=linear_model.LinearRegression(), random_state=30, min_samples=X_train.reshape(-1,1).shape[1]+1)  
reg = model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))  
y_pred = reg.predict(X_test.reshape(-1,1))  
```

Best working linear model RANSACRegressor() with base_estimator LinearRegression(). RANSACRegressor is goot at ignoring outliers.

**TODO**: this has to be implemented, calculate weights based on detecions position.  

#### Polynom fitting

Using Sklearn PolynomialFeatures function to generate X and Y training points for the estimator.  

The PolynomialFeatures and the estimator have to be inputted to the make_pipeline function.  

```python
polyModel = make_pipeline(PolynomialFeatures(degree), linear_model.RANSACRegressor(base_estimator=linear_model.Ridge(alpha=0.5), random_state=30, min_samples=X_train.reshape(-1,1).shape[1]+1))  
polyModel.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))  
y_pred = polyModel.predict(X_test.reshape(-1, 1))  
```

#### Spline

**TODO**: Implement Spline, not working yet.

#### Regression with coordinate depending weigths

Kalman filter calculates velocities, these velocities can be used as weight in the regression.

### Feature extraction, clustering, classification (building a model)

**TODO**: Clustering, KNN <- Scikit Learn  
**Feature extraction -> Clustering**  
**Clustering Algorithm**: Affinity Propagation. (**NOTICE**: This algorithm seems to give nonsense results, will have to test other ones too.)  
**K_MEANS**: Seems to give better results than Affinity Propagation, but still not the results, what we want.  
To make the predictions smarter, a learning algorithm have to be implemented, that trains on the detection and prediction history.  
**NOTICE**: New idea, gather detections, that velocity vector points in the same direction.  
**Feature extraction -> Classification**  

#### Creating the perfect feature vector for clustering

[x, y] the x and y coordinates of the detection  

[x, y, vx, vy] the x, y coordinates and the x, y velocities of the detection  

Not all feature vectors are good for us, there are many false positive detections, that are come from the inaccuracy of yolo. These false positives can be filtered out based on their euclidean distance. Although a threshold value have to be given. The enter and exit points, that distance is under this value, is not chosen as training data for the clustaring algorithm.  

#### Clustering performance evaluation

There are several algorithms that can evaluate the results of our clustering. There are no ground thruth available to us, so only those evaluation algorithms are useful, that require none.  
Scikit-Learn have a few of these: Silhouette Coefficient, Calinski-Harabasz Index, Davies-Bouldin Index. [Clustering performance evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation)


The results of the evaluation algorithms can be displayed with elbow diagrams. There is a python module for this, which is implemented for Scikit-Learns's KMeans algorithm. https://www.scikit-yb.org/en/latest/api/cluster/elbow.html

#### Silhouette Coefficient

Hihger Silhouette Coefficient score realtes to a model with better defined clusters. The Silhouette Coefficient is defined for each sample and is composed of two scores:  

* a: he mean distance between a sample and all other points in the same class.
* b: The mean distance between a sample and all other points in the next nearest cluster.

The Solhouette Coefficient $s$ for a single sample is then given as: $$s = \frac{b - a}{max(a, b)}$$

* The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.
* The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

##### Drawbacks

* The Silhouette Coefficient is generally higher for convex clusters than other concepts of clusters.

##### References

* Peter J. Rousseeuw (1987). [“Silhouettes: a Graphical Aid to the Interpretation and Validation of Cluster Analysis”](https://doi.org/10.1016/0377-0427(87)90125-7). Computational and Applied Mathematics 20: 53–65.

#### Calinski-Harabasz Index

Known as the Variance Ratio Criterion - can be used to evaluate the model, where a higher Calinski-Harabasz score relates to a model with better defined clusters.  
The index is the ratio of the sum of between-clusters dispersion and of whithin-cluster dispersion for all clusters (where dispersion is defined as the sum of distances squared).

* The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
* The score is fast to compute.

##### Drawbacks

* The Calinski-Harabasz index is generally higher for convex clusters than other concepts of clusters.

##### The Math

For a set of data $E$ of size $n_E$ which has been clustered into $k$ clusters, the Calinski-Harabasz score $s$ is defined as the ratio of the between-clusters dispersion mean and the within-cluster dispersion: $$s = \frac{\mathrm{tr}(B_k)}{\mathrm{tr}(W_k)} \times \frac{n_E - k}{k - 1}$$ where $tr(B_k)$ is trace of the between group dispersion matrix and $tr(W_k)$ is ther tace of the within-cluster dispersion matrix defined by: $$W_k = \sum_{q=1}^k \sum_{x \in C_q} (x - c_q) (x - c_q)^T$$ $$B_k = \sum_{q=1}^k n_q (c_q - c_E) (c_q - c_E)^T$$ with $C_q$ the set of points in cluster $q$, $c_q$ the center of cluster $q$, $c_E$ the center of $E$,, and $n_q$ the number of points in cluster $q$.  

##### References

* Caliński, T., & Harabasz, J. (1974). [“A Dendrite Method for Cluster Analysis”](https://www.researchgate.net/publication/233096619_A_Dendrite_Method_for_Cluster_Analysis). Communications in Statistics-theory and Methods 3: 1-27.

#### Davies-Bouldin Index

The Davies-Bouldin index can be used to evaluate the model, where a lower Davies-Bouldin index relates to a model with better separation between the clusters.  
This index signifies the average ‘similarity’ between clusters, where the similarity is a measure that compares the distance between clusters with the size of the clusters themselves.  
Zero is the lowest possible score. Values closer to zero indicate a better partition.  

* The computation of Davies-Bouldin is simpler than that of Silhouette scores.
* The index is solely based on quantities and features inherent to the dataset as its computation only uses point-wise distances.

##### Drawbacks

* The Davies-Boulding index is generally higher for convex clusters than other concepts of clusters.
* The usage of centroid distance limits the distance metric to Euclidean space.

##### The Math

The index is defined as the average similarity between each cluster $C_i$ for $i=1,...,k$ and its most similar one $C_j$. In the context of this index, smilarity is defined as a measure $R_ij$ that trades off:

* $s_i$ the average distance between each points of cluster $i$ and the cetroid of that cluster - also known as cluster diameter.
* $d_ij$ the distance between cluster centroids $i$ and $j$.

A simple choice to construct $R_ij$ so that it is nonnegative and symmetric is: $$R_{ij} = \frac{s_i + s_j}{d_{ij}}$$
Then the Davies-Bouldin index is defined as: $$DB = \frac{1}{k} \sum_{i=1}^k \max_{i \neq j} R_{ij}$$

##### References

* Davies, David L.; Bouldin, Donald W. (1979). [“A Cluster Separation Measure”](https://doi.org/10.1109/TPAMI.1979.4766909) IEEE Transactions on Pattern Analysis and Machine Intelligence. PAMI-1 (2): 224-227.
* Halkidi, Maria; Batistakis, Yannis; Vazirgiannis, Michalis (2001). [“On Clustering Validation Techniques”](https://doi.org/10.1023/A:1012801612483) Journal of Intelligent Information Systems, 17(2-3), 107-145.
* [Wikipedia entry for Davies-Bouldin index](https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index).

## Documentation

1. Building main loop of the program to be able to input video sources, using OpenCV VideoCapture. From VideoCapture object frames can be read. `cv.imshow("FRAME", frame)` imshow function opens GUI window to show actual frame.

```python
    cap = cv.VideoCapture(input)
    # exit if video cant be opened
    if not cap.isOpened():
        print("Source cannot be opened.")
        exit(0)
    .
    .
    .
    while(1):
      ret, frame = cap.read()
      if frame is None:
          break
    
    cv.imshow("FRAME", frame)
    if cv.waitKey(1) == ord('p'):
        if cv.waitKey(0) == ord('r'):
            continue
    if cv.waitKey(10) == ord('q'):
            break
```

1. Implement YOLO API - hldnapi.py - that works with the C-API of Darknet. In this function, the image has to be transformed to Darknet be able to run inference on it. `cv.cvtColor(image, cv.COLOR_BGR2RGB)` convert OpenCV color (Blue,Green,Red) to Darknet color (Red, Green, Blue). `cv.resize(image_rgb, (darknet_width, darknet_height), interpolation=cv.INTER_LINEAR)` resize image to Darknet's neural net image size. `darknet.detect_image(network, class_name, img_for_detect)` run detection on preprocessed image. This function returns a tuple (label, confidence, bbox[x,y,w,h]), the bounding box coordinates have to be resized to the original image.

```python
    def cvimg2detections(image):
        """Fcuntion to make it easy to use darknet with opencv

        Args:
            image (Opencv image): input image to run darknet on

        Returns:
            detections(tuple): detected objects on input image (label, confidence, bbox(x,y,w,h))
        """
        # Convert frame color from BGR to RGB
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        # Resize image for darknet
        image_resized = cv.resize(image_rgb, (darknet_width, darknet_height), interpolation=cv.INTER_LINEAR)
        # Create darknet image
        img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
        # Convert cv2 image to darknet image format
        darknet.copy_image_from_bytes(img_for_detect, image_resized.tobytes())
        # Load image into nn and get detections
        detections = darknet.detect_image(network, class_names, img_for_detect)
        darknet.free_image(img_for_detect)
        # Resize bounding boxes for original frame
        detections_adjusted = []
        for label, confidence, bbox in detections:
            bbox_adjusted = convert2original(image, bbox)
            detections_adjusted.append((str(label), confidence, bbox_adjusted))
        return detections_adjusted
```

3. Implement classes for storing the detections and object trackings. The classes dont have to be overly complex, they must be easy to read and understand. A `class Detection()` and a `class TrackedObject()` was created. The implementation can be found in the historyClass.py file. Detection class has 7 attributes, label, confidence, X, Y, Width, Height, frameID. TrackedObject class has 11, objID, label, futureX, futureY, history, isMoving, time_since_update, max_age, mean, X, Y, VX, VY.

4. Iplement object tracking algorithm. Base idea was to calculate x and y coordinate distances between detection objects. This is a very primitive way of tracking, for initial testing it was good, but I had to find a more accurate tracking algorithm.

5. First prediction algorithm with scikit-learn's LinearRegression function library.  The predictLinear() function takes 3 arguments, a trackedObject object from historyClass.py, historyDepth to determine, how big is the learning set, and a futureDepth to know how far in the future to predict. To do the regression, at least 3 detections should occur. With variable $k$ we can tell the LinearRegression algorithm, on how many points from the training set to train on. Before running the regression, the movementIsRight() function determines wheter the object moving right or left, this is crucial in generation of the prediction points. After we run the regression, the futureX and futureY vector of the trackedObject object can be updated with the predicted values. For the regression I use the simple Ordinary Least Squares (OLS) method. Linear regression formula: $$\hat{y} (w, x) = w_0 + w_1 x_1 + ... + w_p x_p$$ Ordinary Least Squares formula: $$\min_{w} || X w - y||_2^2$$

```python
def movementIsRight(obj: TrackedObject):
    """Returns true, if the object moving right, false otherwise. 

    Args:
        obj (TrackedObject): tracking data of an object 
    
    Return:
        bool: Tru if obj moving right.

    """
    return obj.VX > 0 
    
def predictLinear(trackedObject: TrackedObject, k=3, historyDepth=3, futureDepth=30):
    """Fit linear function on detection history of an object, to predict future coordinates.

    Args:
        trackedObject (TrackedObject): The object, which's future coordinates should be predicted. 
        k (int, optional): Number of training points, ex.: if historyDepth is 30 and k is 3, then the 1st, 15th and 30th points will be training points. Defaults to 3.
        historyDepth (int, optional): Training history length. Defaults to 3.
        futureDepth (int, optional): Prediction vectors length. Defaults to 30.
    """
    x_history = [det.X for det in trackedObject.history]
    y_history = [det.Y for det in trackedObject.history]
    if len(x_history) >= 3 and len(y_history) >= 3:
        # k (int) : number of training points
        # k = len(trackedObject.history) 
        # calculating even slices to pick k points to fit linear model on
        slice = len(trackedObject.history) // k
        X_train = np.array([x for x in x_history[-historyDepth:-1:slice]])
        y_train = np.array([y for y in y_history[-historyDepth:-1:slice]])
        # check if the movement is right or left, because the generated x_test vector
        # if movement is right vector is ascending, otherwise descending
        if movementIsRight(trackedObject):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        # fit linear model on the x_train vectors points
        model = linear_model.LinearRegression(n_jobs=-1)
        reg = model.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
        y_pred = reg.predict(X_test.reshape(-1,1))
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred
```

6. Integrating Deep-SORT tracking into the program. Kalman filter and CNN that has been trained to discriminate pedestrians on a large-scale person re-identification dataset. [[3]](#3) The Kalman filter implementation uses 8 dimensional space (x, y, a, h, vx, vy, va, vh) to track objects.

7. Prediction with Polynom fitting using Scikit-Learn's PolynomTransformer. This is similar to the Linear fitting, but this makes it possible to predict curves in an objects trajectory based on the object's position history. The only difference between the predictLinear and this algorithm, that a PolynomTransformer transforms the history data.

```python
def predictPoly(trackedObject: TrackedObject, degree=3, k=3, historyDepth=3, futureDepth=30):
    """Fit polynomial function on detection history of an object, to predict future coordinates.

    Args:
        trackedObject (TrackedObject): The object, which's future coordinates should be predicted. 
        degree (int, optional): The polynomial functions degree. Defaults to 3.
        k (int, optional): Number of training points, ex.: if historyDepth is 30 and k is 3, then the 1st, 15th and 30th points will be training points. Defaults to 3.
        historyDepth (int, optional): Training history length. Defaults to 3.
        futureDepth (int, optional): Prediction vectors length. Defaults to 30.
    """
    x_history = [det.X for det in trackedObject.history]
    y_history = [det.Y for det in trackedObject.history]
    if len(x_history) >= 3 and len(y_history) >= 3:
        # k (int) : number of training points
        # k = len(trackedObject.history) 
        # calculating even slices to pick k points to fit linear model on
        slice = len(trackedObject.history) // k
        X_train = np.array([x for x in x_history[-historyDepth:-1:slice]])
        y_train = np.array([y for y in y_history[-historyDepth:-1:slice]])
        # generating future points
        if movementIsRight(trackedObject):
            X_test = np.linspace(X_train[-1], X_train[-1]+futureDepth)
        else:
            X_test = np.linspace(X_train[-1], X_train[-1]-futureDepth)
        # poly features
        polyModel = make_pipeline(PolynomialFeatures(degree), linear_model.Ridge(alpha=1e-3))
        polyModel.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))
        # print(X_train.shape, y_train.shape)
        y_pred = polyModel.predict(X_test.reshape(-1, 1))
        trackedObject.futureX = X_test
        trackedObject.futureY = y_pred
```

8. Prediction with splines using Scikit-Learn's SplineTransformer. Spline can only be fitted on data we have, so it cant predict on its own. Before fitting spline on any data, polynom fitting should be done first, then on the result data we can fit a spline curve.

```python
# TODO
```

9. Implement database logging, to save results for later analyzing. The init_db(video_name: str) function creates the database. Name of the video, that is being played, will be the name of the database with a .db appended at the end of it. After the database file is created, schema script will be executed.
This is the schema of the database.

```SQL
CREATE TABLE IF NOT EXISTS objects (
    objID INTEGER PRIMARY KEY NOT NULL,
    label TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS detections (
                objID INTEGER NOT NULL,
                frameNum INTEGER NOT NULL,
                confidence REAL NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                vx REAL NOT NULL,
                vy REAL NOT NULL,
                ax REAL NOT NULL,
                ay REAL NOT NULL,
                FOREIGN KEY(objID) REFERENCES objects(objID)
            );
CREATE TABLE IF NOT EXISTS predictions (
                objID INTEGER NOT NULL,
                frameNum INTEGER NOT NULL,
                idx INTEGER NOT NULL,
                x REAL NOT NULL,
                y REAL NOT NULL
            );
CREATE TABLE IF NOT EXISTS metadata (
                historyDepth INTEGER NOT NULL,
                futureDepth INTEGER NOT NULL,
                yoloVersion TEXT NOT NULL,   
                device TEXT NOT NULL,
                imgsize INTEGER NOT NULL,
                stride INTEGER NOT NULL,
                confidence_threshold REAL NOT NULL,
                iou_threshold REAL NOT NULL
            );
CREATE TABLE IF NOT EXISTS regression (
                linearFunction TEXT NOT NULL,
                polynomFunction TEXT NOT NULL,
                polynomDegree INTEGER NOT NULL,
                trainingPoints INTEGER NOT NULL
);
```

Every object is stored in the objects table, objID as primary key, will help us identify detections. Detections are stored in the detections table, here the objID is a foreign key, that tells us which detection belongs to which object. Predictions have an own table, to a single frame and a single object there can be multiple predictions. THe program's inner environment is also being logged as metadata, historyDepth is the length of the training set. FutureDepth is the length of the prediction vector. Yolo version is also being logged, because of the legacy version 4 (although yolov4 is not really used anymore, it is just an option, that propably will be taken out), imgsize is the input image size of the neural network, stride is how many pixels the convolutonal filter slides over the image. Confidence threshold and iou threshold will determine which detection of yolo will we accept, if the propability of a detection being right. To the regression table, will be the regression function's configuration values stored.  

10. The logging makes it possible, to analyze the data without running the videos each time. For this, data loading functions are needed, that fetches the resutls from the database. These functions are implemented in the databaseLoader.py script. Each function returns a list of all entries logged in the database.

11. Next step after data loading module, is to create heatmap of the traffic data logged from videos. For better visuals, each object has its own coloring, so it also shows, how good DeepSort algorithm works.  

12. With scikit-learn's clustering module, clusters from the gathered data can be created. The point of this, is when a crossroad being observed, the paths can be identified, with this knowledge, personalised training can be done for each scenario. For first k_means algorithm was tested. The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares (see below). This algorithm requires the number of clusters to be specified. It scales well to large numbers of samples and has been used across a large range of application areas in many different fields. The k-means algorithm divides a set of $N$ samples $X$ into $K$ disjoint clusters $C$, each described by the mean of the samples in the cluster. The means are commonly called the cluster “centroids”; note that they are not, in general, points from $X$, although they live in the same space. The K-means algorithm aims to choose centroids that minimise the inertia, or within-cluster sum-of-squares criterion: $$\sum_{i=0}^{n}\min_{\mu_j \in C}(||x_i - \mu_j||^2)$$ Although this algorithm does not require that much computation, cant identify lanes on a crossroad. The result plots can be found in dir "research_data/sherbrooke_video/".  

<figure>
    <img src="research_data/sherbrooke_video/sherbrooke_video_kmeans_n_cluster_2.png" alt="Result of k_means with n_clusters = 2">
    <figcaption align="center">k_means algorithm with 2 initial cluster</figcaption>
</figure>
<figure>
    <img src="research_data/sherbrooke_video/sherbrooke_video_kmeans_n_cluster_3.png" alt="Result of k_means with n_clusters = 3">
    <figcaption align="center">k_means algorithm with 3 initial cluster</figcaption>
</figure>
<figure>
    <img src="research_data/sherbrooke_video/sherbrooke_video_kmeans_n_cluster_4.png" alt="Result of k_means with n_clusters = 3">
    <figcaption align="center">k_means algorithm with 4 initial cluster</figcaption>
</figure>

13.  Using clustering on all detection data, seems to be pointsless, the algorithms cant discriminate different directions from each other. A better approach would be, creating feature vectors from trajectories, then run the clustering algorithms on the extracted features.  

14.  Slow loggin problem solved, big improvement in speed. Creation of shell scripts, which enables sqlite3's Write-Ahead Logging. 4x - 6x times speed improvement. Best solution to slow runtime is to implement a buffer, that stores all detections and predictions, then log them at the end before exiting.  

15.  First try at feature extraction and clustering. First I chose Affinity Propagation Clustring algorithm. AffinityPropagation creates clusters by sending messages between pairs of samples until convergence. A dataset is then described using a small number of exemplars, which are identified as those most representative of other samples. The messages sent between pairs represent the suitability for one sample to be the exemplar of the other, which is updated in response to the values from other pairs. This updating happens iteratively until convergence, at which point the final exemplars are chosen, and hence the final clustering is given. Affinity Propagation can be interesting as it chooses the number of clusters based on the data provided. For this purpose, the two important parameters are the preference, which controls how many exemplars are used, and the damping factor which damps the responsibility and availability messages to avoid numerical oscillations when updating these messages. Algorithm description: The messages sent between points belong to one of two categories. The first is the responsibility $r(i, k)$, which is the accumulated evidence that sample $k$ should be the exemplar for sample $i$. The second is the availability $a(i, k)$ which is the accumulated evidence that sample $i$ should choose $k$ sample to be its exemplar, and considers the values for all other samples that $k$ should be an exemplar. In this way, exemplars are chosen by samples if they are (1) similar enough to many samples and (2) chosen by many samples to be representative of themselves.  More formally, the responsibility of a sample $k$ to be the exemplar of sample $i$ is given by: $$r(i, k) \leftarrow s(i, k) - max [ a(i, k') + s(i, k') \forall k' \neq k ]$$  
Where $s(i, k)$ is the similarity between samples $i$ and $k$. The availability of sample $k$ to be the exemplar of sample $i$ is given by: $$a(i, k) \leftarrow min [0, r(k, k) + \sum_{i'~s.t.~i' \notin \{i, k\}}{r(i', k)}]$$  
To begin with, all values for $r$ and $a$ are set to zero, and the calculation of each iterates until convergence. As discussed above, in order to avoid numerical oscillations when updating the messages, the damping factor $\lambda$ is introduced to iteration process: $$r_{t+1}(i, k) = \lambda\cdot r_{t}(i, k) + (1-\lambda)\cdot r_{t+1}(i, k)$$ $$a_{t+1}(i, k) = \lambda\cdot a_{t}(i, k) + (1-\lambda)\cdot a_{t+1}(i, k)$$ where $t$ indicates the iteration times.  

16. Although affinity propagation does not require initial cluster number, it seems that the results are not usable, because it finds too meny clusters. Other algorithm should be tested ex.: K-Mean, Spectral. For better results, detections should be filtered out, because of false positive detections. Standing objects were detected, so those should be filtered out. The algorithm to filter out only the best data to run clustering on is based on the euclidean distance between enter and exit point pairs. $$d(p,q) = \sqrt{\sum_{i=1}^{n}{(p_i - q_i)^2}}$$

 
<figure>
    <img src="research_data/0005_2_36min/0005_2_36min_affinity_propagation_featureVectors_n_clusters_18_threshold_0.4.png">
    <figcaption align="center">Result of affinity propagation on video 0005_2_36min.mp4</figcaption>
<figure>

1.  Kmeans and Spectral clustering give far better results with the filtered detections, than affinity propagation. Here are the results on the 0005_2_36min.mp4 video.

<figure>
    <img src="research_data/0005_2_36min/0005_2_36min_kmeans_on_nx4_n_cluster_4_threshold_0.6.png">
    <figcaption align="center">Result of kmeans clustering on 0005_2_36min.mp4</figcaption>
</figure>
<figure>
    <img src="research_data/0005_2_36min/0005_2_36min_spectral_on_nx4_n_cluster_4_threshold_0.6.png">
    <figcaption align="center">Result of spectral clustering on 0005_2_36min.mp4</figcaption>
</figure>

17. Finding the optimal number of clusters is very important step to be able to build and train a model. There are many algorithms to evaluate results of clusterings. Also from the evaluation results, we have to tell, which evaluation score is the most optimal, for this, the elbow diagram will give the guidance to find it.

18. To be able to run evaluation algorithm on kmeans clustering results, detections have to be assinged to object tracks. That is an easy task, when there are not many objects and detections in the database, but when 27000 objects and 300000 detections in there, things can go very bad, even if multiprocessing is involved, although I implemented multiprocessing into the algorithm, it wasnt worth it. The solution is to do preprocessing on the data, that means, doing the assignment in the SQL queries.

## References

### Darknet-YOLO

<a id="1">[1]</a>  
@misc{bochkovskiy2020yolov4,  
      title={YOLOv4: Optimal Speed and Accuracy of Object Detection},  
      author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},  
      year={2020},  
      eprint={2004.10934},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV}  
}  
@InProceedings{Wang_2021_CVPR,  
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},  
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},  
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
    month     = {June},  
    year      = {2021},  
    pages     = {13029-13038}  
}

<a id="2">[2]</a>
@article{wang2022yolov7,  
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},  
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},  
  journal={arXiv preprint arXiv:2207.02696},  
  year={2022}  
}  

### DeepSORT

<a id="3">[3]</a>  
@inproceedings{Wojke2017simple,  
  title={Simple Online and Realtime Tracking with a Deep Association Metric},  
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},  
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},  
  year={2017},  
  pages={3645--3649},  
  organization={IEEE},  
  doi={10.1109/ICIP.2017.8296962}  
}  
@inproceedings{Wojke2018deep,  
  title={Deep Cosine Metric Learning for Person Re-identification},  
  author={Wojke, Nicolai and Bewley, Alex},  
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},  
  year={2018},  
  pages={748--756},  
  organization={IEEE},  
  doi={10.1109/WACV.2018.00087}  
}  
