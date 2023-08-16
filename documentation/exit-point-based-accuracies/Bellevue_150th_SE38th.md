# Bellevue 150th SE38th 24 hour dataset training data

## SVM

python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_150th_SE38th_24h/Preprocessed/ --test 0.5 --threshold 0.7 --preprocessed --min-samples 100 --max-eps 0.15 --mse 0.2 --models SVM
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
1935
177
1843
30
715
1433
2232
1914
1169
1749
446
346
1882
60
297
2203
231
1304
364
980
71
415
125
Dataset loaded in 325 s
Number of tracks: 21921
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21921/21921 [00:00<00:00, 181259.14it/s]
Shape of feature vectors: (21921, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10 11 12]
Number of labeled trajectories after clustering: 21469
Clustering done in 49 s
Reduce labels: 21469it [00:00, 709929.93it/s]
Clustered exit centroids: [2 1 2 4 3 0 0 1 0 3 4 3 1]
Exit points clusters: [0 1 2 3 4]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 10734
Size of testing set: 10735
Feature vectors generated in 1 s
Classifier SVM trained in 153 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.610717, top-1: 0.775906, top-2: 0.895753, top-3: 0.952738
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.707545, top-1: 0.798084, top-2: 0.919048, top-3: 0.969056
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.775906 | 0.895753 | 0.952738 |            0.610717 |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.798084 | 0.919048 | 0.969056 |            0.707545 |

## DT

python3 -m sklearnex classification.py --n-jobs 1 exitpoint-metrics --dataset research_data/Bellevue_150th_SE38th_24h/Preprocessed/ --test 0.5 --threshold 0.7 --preprocessed --min-samples 100 --max-eps 0.15 --mse 0.2 --models DT
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
1935
177
1843
30
715
1433
2232
1914
1169
1749
446
346
1882
60
297
2203
231
1304
364
980
71
415
125
Dataset loaded in 311 s
Number of tracks: 21921
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21921/21921 [00:00<00:00, 185654.85it/s]
Shape of feature vectors: (21921, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10 11 12]
Number of labeled trajectories after clustering: 21469
Clustering done in 12 s
Reduce labels: 21469it [00:00, 700159.49it/s]
Clustered exit centroids: [0 3 0 4 2 1 1 3 1 2 4 2 3]
Exit points clusters: [0 1 2 3 4]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 10734
Size of testing set: 10735
Feature vectors generated in 1 s
Classifier DT trained in 4 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.904570, top-1: 0.907983, top-2: 0.929044, top-3: 0.929379
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.907395, top-1: 0.915536, top-2: 0.931215, top-3: 0.931247
|    |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:---|---------:|---------:|---------:|--------------------:|
| DT | 0.907983 | 0.929044 | 0.929379 |             0.90457 |
|    |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:---|---------:|---------:|---------:|--------------------:|
| DT | 0.915536 | 0.931215 | 0.931247 |            0.907395 |


## KNN

python3 -m sklearnex classification.py --n-jobs 1 exitpoint-metrics --dataset research_data/Bellevue_150th_SE38th_24h/Preprocessed/ --test 0.5 --threshold 0.7 --preprocessed --min-samples 100 --max-eps 0.15 --mse 0.2 --models KNN
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
1935
177
1843
30
715
1433
2232
1914
1169
1749
446
346
1882
60
297
2203
231
1304
364
980
71
415
125
Dataset loaded in 317 s
Number of tracks: 21921
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21921/21921 [00:00<00:00, 160016.11it/s]
Shape of feature vectors: (21921, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10 11 12]
Number of labeled trajectories after clustering: 21469
Clustering done in 13 s
Reduce labels: 21469it [00:00, 690670.23it/s]
Clustered exit centroids: [1 3 1 4 2 0 0 3 0 2 4 2 3]
Exit points clusters: [0 1 2 3 4]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 10734
Size of testing set: 10735
Feature vectors generated in 1 s
Classifier KNN trained in 10 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.920993, top-1: 0.941929, top-2: 0.992080, top-3: 0.992959
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.948333, top-1: 0.951509, top-2: 0.993709, top-3: 0.993869
|     |    Top-1 |   Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|--------:|---------:|--------------------:|
| KNN | 0.941929 | 0.99208 | 0.992959 |            0.920993 |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.951509 | 0.993709 | 0.993869 |            0.948333 |
(yolov7_v2) pecneb@pop-os:/media/pecneb/970evoplus/gitclones/computer_vision_research$ 
