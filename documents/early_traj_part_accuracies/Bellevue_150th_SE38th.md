python3 classification.py --n-jobs 12 exitpoint-metrics --dataset research_data/Bellevue_150th_SE38th_24h/Preprocessed/ --test 0.25 --test-part 0.33 --preprocessed --min-samples 100 --max-eps 0.15 --mse 0.2 --models KNN SVM DT
364
125
71
30
60
177
1169
1843
297
1935
1749
1882
2203
1914
2232
446
415
1433
1304
980
715
346
231
Dataset loaded in 163 s
Number of tracks: 21921
Feature vectors.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21921/21921 [00:00<00:00, 691095.44it/s]
Shape of feature vectors: (21921, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10 11 12]
Number of labeled trajectories after clustering: 21468
Clustering done in 77 s
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
Reduce labels: 21468it [00:00, 4027342.26it/s]
Clustered exit centroids: [3 0 1 4 1 2 3 4 2 3 2 0 1]
Exit points clusters: [0 1 2 3 4]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 16101
Size of testing set: 5367
Feature vectors generated in 0 s
Classifier KNN trained in 3 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.940015, top-1: 0.935599, top-2: 0.992682, top-3: 0.993007
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.931122, top-1: 0.936412, top-2: 0.993495, top-3: 0.993658
Classifier SVM trained in 266 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.763251, top-1: 0.823711, top-2: 0.962270, top-3: 0.989429
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.756219, top-1: 0.827452, top-2: 0.963897, top-3: 0.990730
Classifier DT trained in 0 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.895088, top-1: 0.903236, top-2: 0.925028, top-3: 0.928281
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.893019, top-1: 0.903236, top-2: 0.924053, top-3: 0.924053

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.935599 | 0.992682 | 0.993007 |            0.940015 |
| SVM | 0.823711 | 0.96227  | 0.989429 |            0.763251 |
| DT  | 0.903236 | 0.925028 | 0.928281 |            0.895088 |

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.936412 | 0.993495 | 0.993658 |            0.931122 |
| SVM | 0.827452 | 0.963897 | 0.99073  |            0.756219 |
| DT  | 0.903236 | 0.924053 | 0.924053 |            0.893019 |

python3 classification.py --n-jobs 12 exitpoint-metrics --dataset research_data/Bellevue_150th_SE38th_24h/Preprocessed/ --test 0.25 --preprocessed --min-samples 100 --max-eps 0.15 --mse 0.2 --models KNN SVM DT
364
125
71
30
60
177
1169
1843
297
1935
1749
1882
2203
1914
2232
446
415
1433
1304
980
715
346
231
Dataset loaded in 160 s
Number of tracks: 21921
Feature vectors.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21921/21921 [00:00<00:00, 704757.27it/s]
Shape of feature vectors: (21921, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10 11 12]
Number of labeled trajectories after clustering: 21468
Clustering done in 74 s
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
Reduce labels: 21468it [00:00, 4086562.51it/s]
Clustered exit centroids: [1 4 3 2 3 0 1 2 0 1 0 4 3]
Exit points clusters: [0 1 2 3 4]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 16101
Size of testing set: 5367
Feature vectors generated in 0 s
Classifier KNN trained in 10 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.925454, top-1: 0.944800, top-2: 0.993371, top-3: 0.994040
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.949229, top-1: 0.952640, top-2: 0.994646, top-3: 0.994773
Classifier SVM trained in 392 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.789664, top-1: 0.860853, top-2: 0.968416, top-3: 0.993116
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.832687, top-1: 0.875163, top-2: 0.973101, top-3: 0.995283
Classifier DT trained in 1 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.914368, top-1: 0.919750, top-2: 0.935112, top-3: 0.938171
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.917336, top-1: 0.921407, top-2: 0.936801, top-3: 0.936865

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.9448   | 0.993371 | 0.99404  |            0.925454 |
| SVM | 0.860853 | 0.968416 | 0.993116 |            0.789664 |
| DT  | 0.91975  | 0.935112 | 0.938171 |            0.914368 |

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.95264  | 0.994646 | 0.994773 |            0.949229 |
| SVM | 0.875163 | 0.973101 | 0.995283 |            0.832687 |
| DT  | 0.921407 | 0.936801 | 0.936865 |            0.917336 |
