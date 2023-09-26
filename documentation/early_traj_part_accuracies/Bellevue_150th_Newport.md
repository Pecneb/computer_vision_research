 python3 classification.py --n-jobs 12 exitpoint-metrics --dataset research_data/Bellevue_150th_Newport_24h_v2/Preprocessed/ --test 0.25 --test-part 0.33 --preprocessed --min-samples 200 --max-eps 0.1 --mse 0.2 --models KNN SVM DT
3224
1832
1301
1149
722
301
204
111
52
22
60
159
404
1209
4492
2737
2208
1925
2106
1630
2439
4009
4760
5106
Dataset loaded in 51 s
Number of tracks: 42162
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42162/42162 [00:00<00:00, 776527.87it/s]
Shape of feature vectors: (42162, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10]
Number of labeled trajectories after clustering: 39326
Clustering done in 149 s
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
Reduce labels: 39326it [00:00, 4163390.36it/s]
Clustered exit centroids: [1 0 3 1 2 3 1 0 2 2 3]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 29494
Size of testing set: 9832
Feature vectors generated in 0 s
Classifier KNN trained in 5 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.853362, top-1: 0.898087, top-2: 0.983914, top-3: 0.990289
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.876443, top-1: 0.900736, top-2: 0.984600, top-3: 0.990289
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/svm/_base.py:299: ConvergenceWarning: Solver terminated early (max_iter=26000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
  warnings.warn(
Classifier SVM trained in 1148 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.543510, top-1: 0.674939, top-2: 0.920059, top-3: 0.995782
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.475174, top-1: 0.677391, top-2: 0.928298, top-3: 0.998431
Classifier DT trained in 2 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.827411, top-1: 0.850515, top-2: 0.886121, top-3: 0.899951
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.836986, top-1: 0.855910, top-2: 0.882786, top-3: 0.883178

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.898087 | 0.983914 | 0.990289 |            0.853362 |
| SVM | 0.674939 | 0.920059 | 0.995782 |            0.54351  |
| DT  | 0.850515 | 0.886121 | 0.899951 |            0.827411 |

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.900736 | 0.9846   | 0.990289 |            0.876443 |
| SVM | 0.677391 | 0.928298 | 0.998431 |            0.475174 |
| DT  | 0.85591  | 0.882786 | 0.883178 |            0.836986 |

python3 classification.py --n-jobs 12 exitpoint-metrics --dataset research_data/Bellevue_150th_Newport_24h_v2/Preprocessed/ --test 0.25 --preprocessed --min-samples 200 --max-eps 0.1 --mse 0.2 --models KNN SVM DT
3224
1832
1301
1149
722
301
204
111
52
22
60
159
404
1209
4492
2737
2208
1925
2106
1630
2439
4009
4760
5106
Dataset loaded in 55 s
Number of tracks: 42162
Feature vectors.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42162/42162 [00:00<00:00, 774175.42it/s]
Shape of feature vectors: (42162, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10]
Number of labeled trajectories after clustering: 39326
Clustering done in 161 s
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
Reduce labels: 39326it [00:00, 3887558.02it/s]
Clustered exit centroids: [3 2 0 3 1 0 3 2 1 1 0]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 29494
Size of testing set: 9832
Feature vectors generated in 0 s
Classifier KNN trained in 17 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.880720, top-1: 0.931317, top-2: 0.986339, top-3: 0.990924
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.928693, top-1: 0.943008, top-2: 0.987893, top-3: 0.990981
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/svm/_base.py:299: ConvergenceWarning: Solver terminated early (max_iter=26000).  Consider pre-processing your data with StandardScaler or MinMaxScaler.
  warnings.warn(
Classifier SVM trained in 1452 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.669260, top-1: 0.773679, top-2: 0.943860, top-3: 0.993994
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.753756, top-1: 0.841187, top-2: 0.965384, top-3: 0.998674
Classifier DT trained in 2 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.873278, top-1: 0.888933, top-2: 0.918888, top-3: 0.927035
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.895468, top-1: 0.902044, top-2: 0.919400, top-3: 0.919494

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.931317 | 0.986339 | 0.990924 |            0.88072  |
| SVM | 0.773679 | 0.94386  | 0.993994 |            0.66926  |
| DT  | 0.888933 | 0.918888 | 0.927035 |            0.873278 |

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.943008 | 0.987893 | 0.990981 |            0.928693 |
| SVM | 0.841187 | 0.965384 | 0.998674 |            0.753756 |
| DT  | 0.902044 | 0.9194   | 0.919494 |            0.895468 |