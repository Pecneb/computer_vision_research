python3 classification.py --n-jobs 12 exitpoint-metrics --dataset research_data/Bellevue_Eastgate_24h/Preprocessed/ --test 0.25 --test-part 0.33 --preprocessed --min-samples 200 --max-eps 0.16 --mse 0.2 --models KNN SVM DT
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
Dataset loaded in 174 s
Number of tracks: 32261
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 597308.35it/s]
Shape of feature vectors: (32261, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8]
Number of labeled trajectories after clustering: 32052
Clustering done in 92 s
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
Reduce labels: 32052it [00:00, 3341515.01it/s]
Clustered exit centroids: [2 1 0 3 2 0 3 0 1]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 24039
Size of testing set: 8013
Feature vectors generated in 0 s
Classifier KNN trained in 3 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.953816, top-1: 0.973698, top-2: 0.993799, top-3: 0.996579
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.970856, top-1: 0.976799, top-2: 0.994975, top-3: 0.995082
Classifier SVM trained in 544 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.926354, top-1: 0.954239, top-2: 0.993799, top-3: 0.997220
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.942179, top-1: 0.958195, top-2: 0.997541, top-3: 0.999893
Classifier DT trained in 1 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.937588, top-1: 0.962900, top-2: 0.973164, top-3: 0.984390
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.947226, top-1: 0.960975, top-2: 0.968994, top-3: 0.968994

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.973698 | 0.993799 | 0.996579 |            0.953816 |
| SVM | 0.954239 | 0.993799 | 0.99722  |            0.926354 |
| DT  | 0.9629   | 0.973164 | 0.98439  |            0.937588 |

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.976799 | 0.994975 | 0.995082 |            0.970856 |
| SVM | 0.958195 | 0.997541 | 0.999893 |            0.942179 |
| DT  | 0.960975 | 0.968994 | 0.968994 |            0.947226 |

python3 classification.py --n-jobs 12 exitpoint-metrics --dataset research_data/Bellevue_Eastgate_24h/Preprocessed/ --test 0.25 --preprocessed --min-samples 200 --max-eps 0.16 --mse 0.2 --models KNN SVM DT
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
Dataset loaded in 175 s
Number of tracks: 32261
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 654933.04it/s]
Shape of feature vectors: (32261, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8]
Number of labeled trajectories after clustering: 32052
Clustering done in 82 s
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
Reduce labels: 32052it [00:00, 3348339.52it/s]
Clustered exit centroids: [2 1 0 3 2 0 3 0 1]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 24039
Size of testing set: 8013
Feature vectors generated in 0 s
Classifier KNN trained in 13 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.955299, top-1: 0.967296, top-2: 0.995294, top-3: 0.997882
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.984205, top-1: 0.987274, top-2: 0.997241, top-3: 0.997476
Classifier SVM trained in 699 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.913096, top-1: 0.941437, top-2: 0.992984, top-3: 0.998631
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.975858, top-1: 0.981841, top-2: 0.998118, top-3: 0.999957
Classifier DT trained in 1 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.964497, top-1: 0.973970, top-2: 0.981755, top-3: 0.989605
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.970757, top-1: 0.976194, top-2: 0.980344, top-3: 0.980344

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.967296 | 0.995294 | 0.997882 |            0.955299 |
| SVM | 0.941437 | 0.992984 | 0.998631 |            0.913096 |
| DT  | 0.97397  | 0.981755 | 0.989605 |            0.964497 |

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.987274 | 0.997241 | 0.997476 |            0.984205 |
| SVM | 0.981841 | 0.998118 | 0.999957 |            0.975858 |
| DT  | 0.976194 | 0.980344 | 0.980344 |            0.970757 |
