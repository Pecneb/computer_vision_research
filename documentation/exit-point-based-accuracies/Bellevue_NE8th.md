# Bellevue NE8th Dataset

## SVM

python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_NE8th_24h_v2/Preprocessed_threshold_0.4/ --test 0.5 --output research_data/Bellevue_NE8th_24h_v2/Preprocessed_threshold_0.4/ --preprocessed --min-samples 200 --max-eps 0.15 --mse 0.2 --models SVM --threshold 0.4 
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
1572
5539
4610
3671
2696
1057
895
571
269
311
429
871
2181
4188
6086
5951
5301
6319
6892
6083
7014
7813
7996
Dataset loaded in 156 s
Number of tracks: 88315
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 88315/88315 [00:00<00:00, 568098.27it/s]
Shape of feature vectors: (88315, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10 11]
Number of labeled trajectories after clustering: 24386
Clustering done in 250 s
Reduce labels: 24386it [00:00, 4788272.90it/s]
Clustered exit centroids: [2 0 0 1 4 4 3 1 2 2 1 2]
Exit points clusters: [0 1 2 3 4]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 12193
Size of testing set: 12193
Feature vectors generated in 0 s
Classifier SVM trained in 56 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.901832, top-1: 0.947400, top-2: 0.997200, top-3: 0.999420
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.986325, top-1: 0.983274, top-2: 0.999333, top-3: 0.999797
Killed
