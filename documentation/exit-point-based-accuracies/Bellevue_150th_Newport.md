## KNN

python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_150th_Newport_24h_v2/Preprocessed_threshold_0.4/ --test 0.5 --output research_data/Bellevue_150th_Newport_24h_v2/Preprocessed_threshold_0.4/ --preprocessed --min-samples 200 --max-eps 0.1 --mse 0.2 --models KNN                                                        

Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
3196
1822
1294
1141
719
299
204
111
52
22
60
159
404
1208
4456
2727
2199
1913
2088
1620
2398
3975
4753
5082
Dataset loaded in 46 s
Number of tracks: 41902
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41902/41902 [00:00<00:00, 557252.78it/s]
Shape of feature vectors: (41902, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10]
Number of labeled trajectories after clustering: 39325
Clustering done in 99 s
Reduce labels: 39325it [00:00, 4817202.24it/s]
Clustered exit centroids: [0 2 3 0 1 3 0 2 1 1 3]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 19662
Size of testing set: 19663
Feature vectors generated in 0 s
Classifier KNN trained in 7 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.875785, top-1: 0.927191, top-2: 0.984904, top-3: 0.990047
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.925471, top-1: 0.940307, top-2: 0.986893, top-3: 0.989573
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.927191 | 0.984904 | 0.990047 |            0.875785 |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.940307 | 0.986893 | 0.989573 |            0.925471 |


## SVM

python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_150th_Newport_24h_v2/Preprocessed_threshold_0.4/ --test 0.5 --output research_data/Bellevue_150th_Newport_24h_v2/Preprocessed_threshold_0.4/ --preprocessed --min-samples 200 --max-eps 0.1 --mse 0.2 --models SVM   
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
3196
1822
1294
1141
719
299
204
111
52
22
60
159
404
1208
4456
2727
2199
1913
2088
1620
2398
3975
4753
5082
Dataset loaded in 50 s
Number of tracks: 41902
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41902/41902 [00:00<00:00, 607382.35it/s]
Shape of feature vectors: (41902, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10]
Number of labeled trajectories after clustering: 39325
Clustering done in 130 s
Reduce labels: 39325it [00:00, 4837121.46it/s]
Clustered exit centroids: [3 2 0 3 1 0 3 2 1 1 0]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 19662
Size of testing set: 19663
Feature vectors generated in 0 s
Classifier SVM trained in 127 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.722348, top-1: 0.830876, top-2: 0.956332, top-3: 0.996373
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.843607, top-1: 0.883106, top-2: 0.973312, top-3: 0.999725
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.830876 | 0.956332 | 0.996373 |            0.722348 |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.883106 | 0.973312 | 0.999725 |            0.843607 |
(yolov7) pecneb@rigel:/media/pecneb/4d646cbd-cce0-42c4-bdf5-b43cc196e4a1/gitclones/computer_vision_research$ 


# DT

python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_150th_Newport_24h_v2/Preprocessed_threshold_0.4/ --test 0.5 --output research_data/Bellevue_150th_Newport_24h_v2/Preprocessed_threshold_0.4/ --preprocessed --min-samples 200 --max-eps 0.1 --mse 0.2 --models DT
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
3196
1822
1294
1141
719
299
204
111
52
22
60
159
404
1208
4456
2727
2199
1913
2088
1620
2398
3975
4753
5082
Dataset loaded in 49 s
Number of tracks: 41902
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41902/41902 [00:00<00:00, 632634.74it/s]
Shape of feature vectors: (41902, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10]
Number of labeled trajectories after clustering: 39325
Clustering done in 140 s
Reduce labels: 39325it [00:00, 4940424.27it/s]
Clustered exit centroids: [3 2 0 3 1 0 3 2 1 1 0]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 19662
Size of testing set: 19663
Feature vectors generated in 0 s
Classifier DT trained in 2 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.865413, top-1: 0.881533, top-2: 0.915144, top-3: 0.923677
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.888597, top-1: 0.897870, top-2: 0.916508, top-3: 0.916603
|    |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:---|---------:|---------:|---------:|--------------------:|
| DT | 0.881533 | 0.915144 | 0.923677 |            0.865413 |
|    |   Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:---|--------:|---------:|---------:|--------------------:|
| DT | 0.89787 | 0.916508 | 0.916603 |            0.888597 |
(yolov7) pecneb@rigel:/media/pecneb/4d646cbd-cce0-42c4-bdf5-b43cc196e4a1/gitclones/computer_vision_research$ 
