python3 -m sklearnex classification.py --n-jobs 4 exitpoint-metrics --dataset /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/Preprocessed_threshold_0.7_enter-exit-distance_0.1/ --out /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/Preprocessed_threshold_0.7_enter-exit-distance_0.1/ --test 0.2 --preprocessed --min-samples 100 --max-eps 0.25 --mse 0.2 --models SVM KNN DT                           

Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
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
Dataset loaded in 11 s
Number of tracks: 3836
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3836/3836 [00:00<00:00, 234939.33it/s]
Shape of feature vectors: (3836, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8]
Number of labeled trajectories after clustering: 3348
Clustering done in 7 s
Reduce labels: 3348it [00:00, 727931.67it/s]
Clustered exit centroids: [2 3 0 3 2 1 1 0 1]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 2678
Size of testing set: 670
Feature vectors generated in 0 s
Classifier SVM trained in 16 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.648308, top-1: 0.811653, top-2: 0.893177, top-3: 0.955277
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.803874, top-1: 0.834398, top-2: 0.922566, top-3: 0.981855
Classifier KNN trained in 0 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.914868, top-1: 0.945822, top-2: 0.993355, top-3: 0.995656
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.962568, top-1: 0.966266, top-2: 0.995911, top-3: 0.995911
Classifier DT trained in 0 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.924564, top-1: 0.935599, top-2: 0.954255, top-3: 0.964222
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.950601, top-1: 0.943010, top-2: 0.952466, top-3: 0.952466

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.811653 | 0.893177 | 0.955277 |            0.648308 |
| KNN | 0.945822 | 0.993355 | 0.995656 |            0.914868 |
| DT  | 0.935599 | 0.954255 | 0.964222 |            0.924564 |

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.834398 | 0.922566 | 0.981855 |            0.803874 |
| KNN | 0.966266 | 0.995911 | 0.995911 |            0.962568 |
| DT  | 0.94301  | 0.952466 | 0.952466 |            0.950601 |
