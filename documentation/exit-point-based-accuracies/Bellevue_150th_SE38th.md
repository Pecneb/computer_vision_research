# Bellevue 150th SE38th 24 hour dataset training data

## SVM

python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_150th_SE38th_24h/Preprocessed/ --output research_data/Bellevue_150th_SE38th_24h/Preprocessed/ --preprocessed --min-samples 100 --max-eps 0.2 --models SVM --test 0.5
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
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21921/21921 [00:00<00:00, 259079.25it/s]
Shape of feature vectors: (21921, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9]
Number of labeled trajectories after clustering: 19929
Clustering done in 50 s
Reduce labels: 19929it [00:00, 675777.61it/s]
Clustered exit centroids: [0 0 1 0 1 1 0 1 0 0]
Exit points clusters: [0 1]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 9964
Size of testing set: 9965
Feature vectors generated in 0 s
Classifier SVM trained in 97 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.636723, top-1: 0.788731, top-2: 0.947488, top-3: 0.976650
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.856992, top-1: 0.835723, top-2: 1.000000, top-3: 1.000000
|     |    Top-1 |    Top-2 |   Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|--------:|--------------------:|
| SVM | 0.788731 | 0.947488 | 0.97665 |            0.636723 |
|     |    Top-1 |   Top-2 |   Top-3 |   Balanced Accuracy |
|:----|---------:|--------:|--------:|--------------------:|
| SVM | 0.835723 |       1 |       1 |            0.856992 |

# DecisionTree - DT

(yolov7_v2) pecneb@pop-os:/media/pecneb/970evoplus/gitclones/computer_vision_research$ python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_150th_SE38th_24h/Preprocessed/ --output research_data/Bellevue_150th_SE38th_24h/Preprocessed/ --preprocessed --min-samples 100 --max-eps 0.2 --models DT --test 0.5
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
Dataset loaded in 315 s
Number of tracks: 21921
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21921/21921 [00:00<00:00, 252436.94it/s]
Shape of feature vectors: (21921, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9]
Number of labeled trajectories after clustering: 19929
Clustering done in 49 s
Reduce labels: 19929it [00:00, 700345.06it/s]
Clustered exit centroids: [0 0 1 0 1 1 0 1 0 0]
Exit points clusters: [0 1]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 9964
Size of testing set: 9965
Feature vectors generated in 0 s
Classifier DT trained in 2 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.932591, top-1: 0.933887, top-2: 0.947797, top-3: 0.948193
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.936536, top-1: 0.939991, top-2: 0.949431, top-3: 0.949431
|    |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:---|---------:|---------:|---------:|--------------------:|
| DT | 0.933887 | 0.947797 | 0.948193 |            0.932591 |
|    |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:---|---------:|---------:|---------:|--------------------:|
| DT | 0.939991 | 0.949431 | 0.949431 |            0.936536 |
