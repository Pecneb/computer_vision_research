python3 classification.py --n-jobs 12 exitpoint-metrics --dataset research_data/Bellevue_NE8th_24h_v2/Preprocessed_threshold_0.7_enter-exit-distance_1.0/ --test 0.25 --test-part 0.33 --preprocessed --min-samples 400 --max-eps 0.15 --mse 0.2 --models KNN SVM DT
2851
63
316
2171
889
196
2289
2819
904
206
466
2754
1802
457
2590
87
2343
2161
103
1950
1275
2205
2683
Dataset loaded in 81 s
Number of tracks: 33580
Feature vectors.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33580/33580 [00:00<00:00, 665284.54it/s]
Shape of feature vectors: (33580, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10]
Number of labeled trajectories after clustering: 32025
Clustering done in 86 s
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
Reduce labels: 32025it [00:00, 4015022.74it/s]
Clustered exit centroids: [0 1 1 3 0 2 1 0 3 2 2]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 24018
Size of testing set: 8007
Feature vectors generated in 0 s
Classifier KNN trained in 4 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.920041, top-1: 0.969142, top-2: 0.995931, top-3: 0.996383
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.971964, top-1: 0.972759, top-2: 0.996722, top-3: 0.996722
Classifier SVM trained in 410 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.793439, top-1: 0.932293, top-2: 0.994235, top-3: 0.997739
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.936386, top-1: 0.937719, top-2: 0.999096, top-3: 0.999774
Classifier DT trained in 1 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.907182, top-1: 0.956935, top-2: 0.972646, top-3: 0.974002
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.958762, top-1: 0.957048, top-2: 0.964847, top-3: 0.964847

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.969142 | 0.995931 | 0.996383 |            0.920041 |
| SVM | 0.932293 | 0.994235 | 0.997739 |            0.793439 |
| DT  | 0.956935 | 0.972646 | 0.974002 |            0.907182 |

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.972759 | 0.996722 | 0.996722 |            0.971964 |
| SVM | 0.937719 | 0.999096 | 0.999774 |            0.936386 |
| DT  | 0.957048 | 0.964847 | 0.964847 |            0.958762 |

python3 classification.py --n-jobs 12 exitpoint-metrics --dataset research_data/Bellevue_NE8th_24h_v2/Preprocessed_threshold_0.7_enter-exit-distance_1.0/ --test 0.25 --preprocessed --min-samples 400 --max-eps 0.15 --mse 0.2 --models KNN SVM DT                                                                                                                                                                                                                                                                                
2851
63
316
2171
889
196
2289
2819
904
206
466
2754
1802
457
2590
87
2343
2161
103
1950
1275
2205
2683
Dataset loaded in 83 s
Number of tracks: 33580
Feature vectors.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33580/33580 [00:00<00:00, 678802.49it/s]
Shape of feature vectors: (33580, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10]
Number of labeled trajectories after clustering: 32025
Clustering done in 127 s
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.11/site-packages/sklearn/cluster/_kmeans.py:870: FutureWarning: The default value of `n_init` will change from 10 to 'auto' in 1.4. Set the value of `n_init` explicitly to suppress the warning
  warnings.warn(
Reduce labels: 32025it [00:00, 4174231.19it/s]
Clustered exit centroids: [2 1 1 3 2 0 1 2 3 0 0]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 24018
Size of testing set: 8007
Feature vectors generated in 0 s
Classifier KNN trained in 14 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.921781, top-1: 0.958425, top-2: 0.990811, top-3: 0.997108
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.987585, top-1: 0.987812, top-2: 0.998029, top-3: 0.998029
Classifier SVM trained in 663 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.806652, top-1: 0.920170, top-2: 0.985264, top-3: 0.998372
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.968595, top-1: 0.969221, top-2: 0.999529, top-3: 0.999807
Classifier DT trained in 2 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.930808, top-1: 0.958982, top-2: 0.971705, top-3: 0.972648
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.972179, top-1: 0.970634, top-2: 0.974447, top-3: 0.974447

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.958425 | 0.990811 | 0.997108 |            0.921781 |
| SVM | 0.92017  | 0.985264 | 0.998372 |            0.806652 |
| DT  | 0.958982 | 0.971705 | 0.972648 |            0.930808 |

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| KNN | 0.987812 | 0.998029 | 0.998029 |            0.987585 |
| SVM | 0.969221 | 0.999529 | 0.999807 |            0.968595 |
| DT  | 0.970634 | 0.974447 | 0.974447 |            0.972179 |
