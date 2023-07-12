python3 -m sklearnex classification.py --n_jobs 16 exitpoint-metrics --dataset research_data/Bellevue_Eastgate_24h/Preprocessed/ --output research_data/Bellevue_Eastgate_24h/Preprocessed/ --preprocessed --min-samples 200 --max-eps 0.16 --mse 0.2
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
2429
103
60
29
57
120
357
1216
3137
2593
2158
2268
2573
2053
2517
3206
3321
1000
1154
811
589
197
313
Dataset loaded in 215 s
Number of tracks: 32261
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 583009.58it/s]
Shape of feature vectors: (32261, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8]
Number of labeled trajectories after clustering: 32052
Clustering done in 124 s
Reduce labels: 32052it [00:00, 4407443.18it/s]
Clustered exit centroids: [2 1 0 3 2 0 3 0 1]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 25641
Size of testing set: 6411
Feature vectors generated in 0 s
Classifier SVM trained in 209 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.903399, top-1: 0.937936, top-2: 0.991825, top-3: 0.998691
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.977386, top-1: 0.981271, top-2: 0.997195, top-3: 0.999947
Classifier DT trained in 2 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.964770, top-1: 0.974351, top-2: 0.982447, top-3: 0.989554
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.971779, top-1: 0.976355, top-2: 0.980416, top-3: 0.980470
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.937936 | 0.991825 | 0.998691 |            0.903399 |
| DT  | 0.974351 | 0.982447 | 0.989554 |            0.96477  |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.981271 | 0.997195 | 0.999947 |            0.977386 |
| DT  | 0.976355 | 0.980416 | 0.98047  |            0.971779 |
