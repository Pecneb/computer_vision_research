python3 -m sklearnex classification.py --n-jobs 4 exitpoint-metrics --dataset ../../cv_research_video_dataset/Bellevue_116th_NE12th_24h/Preprocessed_threshold_0.7_enter-exit-distance_0.1/ --test 0.2 --test-part 0.33 --preprocessed --min-samples 100 --max-eps 0.25 --mse 0.2 --models KNN SVM DTIntel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
52
314
22
246
36
25
18
96
706
29
22
749
172
22
29
153
314
190
579
62
Dataset loaded in 9 s
Number of tracks: 3836
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3836/3836 [00:00<00:00, 277622.77it/s]
Shape of feature vectors: (3836, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8]
Number of labeled trajectories after clustering: 3348
Clustering done in 6 s
Reduce labels: 3348it [00:00, 731077.14it/s]
Clustered exit centroids: [3 1 2 1 3 0 0 2 0]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 2678
Size of testing set: 670
Feature vectors generated in 0 s
Classifier KNN trained in 1 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.900611, top-1: 0.925878, top-2: 0.993498, top-3: 0.993498
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.919502, top-1: 0.925878, top-2: 0.993498, top-3: 0.993498
Classifier SVM trained in 11 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.646398, top-1: 0.823147, top-2: 0.927178, top-3: 0.968791
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.808184, top-1: 0.823147, top-2: 0.933680, top-3: 0.977893
Classifier DT trained in 0 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.896244, top-1: 0.908973, top-2: 0.927178, top-3: 0.944083
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.918665, top-1: 0.899870, top-2: 0.921977, top-3: 0.921977

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.925878 | 0.993498 | 0.993498 |            0.900611 |
| SVM | 0.823147 | 0.927178 | 0.968791 |            0.646398 |
| DT  | 0.908973 | 0.927178 | 0.944083 |            0.896244 |

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.925878 | 0.993498 | 0.993498 |            0.919502 |
| SVM | 0.823147 | 0.93368  | 0.977893 |            0.808184 |
| DT  | 0.89987  | 0.921977 | 0.921977 |            0.918665 |
