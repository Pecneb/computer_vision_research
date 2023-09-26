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

python3 -m sklearnex classification.py --n_jobs 16 exitpoint-metrics --dataset research_data/Bellevue_150th_SE38th_24h/Preprocessed/ --preprocessed --min-samples 50 --max-eps 0.1 --mse 0.2
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)
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
Dataset loaded in 199 s
Number of tracks: 21921
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21921/21921 [00:00<00:00, 577196.35it/s]
Shape of feature vectors: (21921, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15]
Number of labeled trajectories after clustering: 20918
Clustering done in 80 s
Reduce labels: 20918it [00:00, 4891367.07it/s]
Clustered exit centroids: [0 1 3 1 3 2 4 0 4 0 2 4 2 1 3 2]
Exit points clusters: [0 1 2 3 4]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 16734
Size of testing set: 4184
Feature vectors generated in 0 s
Classifier SVM trained in 136 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.757384, top-1: 0.861116, top-2: 0.965473, top-3: 0.994225
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.850854, top-1: 0.887205, top-2: 0.976204, top-3: 0.998812
Classifier DT trained in 1 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.898989, top-1: 0.912844, top-2: 0.928408, top-3: 0.928858
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.919018, top-1: 0.919110, top-2: 0.931029, top-3: 0.931070
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.861116 | 0.965473 | 0.994225 |            0.757384 |
| DT  | 0.912844 | 0.928408 | 0.928858 |            0.898989 |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.887205 | 0.976204 | 0.998812 |            0.850854 |
| DT  | 0.91911  | 0.931029 | 0.93107  |            0.919018 |

python3 -m sklearnex classification.py --n-jobs 10 exitpoint-metrics --dataset re
search_data/Bellevue_Eastgate_24h/Preprocessed/ --test 0.6 --output research_data/Bellevue_Eastgate_24h/Preprocessed/ --preprocessed --min-samples 200 --max-eps 0.16 --mse 0.2               
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
Dataset loaded in 213 s                                                                                                                                                                       
Number of tracks: 32261                                                                                                                                                                       
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
██████████████████████| 32261/32261 [00:00<00:00, 563407.45it/s]                                                                                                                              
Shape of feature vectors: (32261, 4)                                                                                                                                                          
Classes: [-1  0  1  2  3  4  5  6  7  8]                                                                                                                                                      
Number of labeled trajectories after clustering: 32052                                                                                                                                        
Clustering done in 112 s                                                                                                                                                                      
Reduce labels: 32052it [00:00, 3924560.85it/s]                                                                                                                                                
Clustered exit centroids: [2 1 0 3 2 0 3 0 1]                                                                                                                                                 
Exit points clusters: [0 1 2 3]                                                                                                                                                               
Exit point clustering done in 0 s                                                                                                                                                             
Train test split done in 0 s                                                                                                                                                                  
Size of training set: 12820                                                                                                                                                                   
Size of testing set: 19232                                                                                                                                                                    
Feature vectors generated in 0 s                                                             
Classifier SVM trained in 62 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.890117, top-1: 0.928758, top-2: 0.991559, top-3: 0.998929
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.971018, top-1: 0.976987, top-2: 0.997091, top-3: 0.999911
Classifier KNN trained in 6 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.952376, top-1: 0.964736, top-2: 0.995342, top-3: 0.998046
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.984397, top-1: 0.987499, top-2: 0.997519, top-3: 0.997707
Classifier DT trained in 0 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.961379, top-1: 0.972142, top-2: 0.980878, top-3: 0.987918
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.969596, top-1: 0.973953, top-2: 0.978129, top-3: 0.978147
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.928758 | 0.991559 | 0.998929 |            0.890117 |
| KNN | 0.964736 | 0.995342 | 0.998046 |            0.952376 |
| DT  | 0.972142 | 0.980878 | 0.987918 |            0.961379 |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.976987 | 0.997091 | 0.999911 |            0.971018 |
| KNN | 0.987499 | 0.997519 | 0.997707 |            0.984397 |
| DT  | 0.973953 | 0.978129 | 0.978147 |            0.969596 |

python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_Eastgate_24h/Preprocessed/ --test 0.5 --output research_data/Bellevue_Eastgate_24h/Preprocessed/ --preprocessed --min-samples 200 --max-eps 0.16 --mse 0.2
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
Dataset loaded in 214 s
Number of tracks: 32261
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 432922.23it/s]
Shape of feature vectors: (32261, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8]
Number of labeled trajectories after clustering: 32052
Clustering done in 79 s
Reduce labels: 32052it [00:00, 4321306.07it/s]
Clustered exit centroids: [1 2 3 0 1 3 0 3 2]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 16026
Size of testing set: 16026
Feature vectors generated in 0 s
Classifier SVM trained in 85 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.897036, top-1: 0.933449, top-2: 0.992106, top-3: 0.998918
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.973783, top-1: 0.978984, top-2: 0.997161, top-3: 0.999914
Classifier KNN trained in 6 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.954269, top-1: 0.965862, top-2: 0.995737, top-3: 0.998254
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.984456, top-1: 0.987628, top-2: 0.997729, top-3: 0.997879
Classifier DT trained in 2 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.962410, top-1: 0.972953, top-2: 0.981651, top-3: 0.988839
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.970972, top-1: 0.975406, top-2: 0.978588, top-3: 0.978598
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.933449 | 0.992106 | 0.998918 |            0.897036 |
| KNN | 0.965862 | 0.995737 | 0.998254 |            0.954269 |
| DT  | 0.972953 | 0.981651 | 0.988839 |            0.96241  |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.978984 | 0.997161 | 0.999914 |            0.973783 |
| KNN | 0.987628 | 0.997729 | 0.997879 |            0.984456 |
| DT  | 0.975406 | 0.978588 | 0.978598 |            0.970972 |

python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_Eastgate_24h/Preprocessed/ --test 0.6 --output research_data/Bellevue_Eastgate_24h/Preprocessed/ --preprocessed --min-samples 200 --max-eps 0.16 --mse 0.2                                                                                                                                                
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
Dataset loaded in 221 s
Number of tracks: 32261
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 515097.25it/s]
Shape of feature vectors: (32261, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8]
Number of labeled trajectories after clustering: 32052
Clustering done in 52 s
Reduce labels: 32052it [00:00, 5109103.17it/s]
Clustered exit centroids: [2 1 0 3 2 0 3 0 1]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 12820
Size of testing set: 19232
Feature vectors generated in 0 s
Classifier SVM trained in 62 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.890256, top-1: 0.928874, top-2: 0.991639, top-3: 0.998938
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.970945, top-1: 0.976960, top-2: 0.997136, top-3: 0.999911
Classifier KNN trained in 5 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.952376, top-1: 0.964736, top-2: 0.995342, top-3: 0.998046
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.984397, top-1: 0.987499, top-2: 0.997519, top-3: 0.997707
Classifier DT trained in 1 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.961318, top-1: 0.971776, top-2: 0.980280, top-3: 0.987401
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.969382, top-1: 0.973677, top-2: 0.977942, top-3: 0.977951
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.928874 | 0.991639 | 0.998938 |            0.890256 |
| KNN | 0.964736 | 0.995342 | 0.998046 |            0.952376 |
| DT  | 0.971776 | 0.98028  | 0.987401 |            0.961318 |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.97696  | 0.997136 | 0.999911 |            0.970945 |
| KNN | 0.987499 | 0.997519 | 0.997707 |            0.984397 |
| DT  | 0.973677 | 0.977942 | 0.977951 |            0.969382 |

python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_Eastgate_24h/Preprocessed/ --test 0.4 --output research_data/Bellevue_Eastgate_24h/Preprocessed/ --preprocessed --min-samples 200 --max-eps 0.16 --mse 0.2
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
Dataset loaded in 214 s
Number of tracks: 32261
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 539142.80it/s]
Shape of feature vectors: (32261, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8]
Number of labeled trajectories after clustering: 32052
Clustering done in 77 s
Reduce labels: 32052it [00:00, 4875279.49it/s]
Clustered exit centroids: [1 2 3 0 1 3 0 3 2]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 19231
Size of testing set: 12821
Feature vectors generated in 0 s
Classifier SVM trained in 114 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.901518, top-1: 0.936212, top-2: 0.991876, top-3: 0.998782
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.976009, top-1: 0.980339, top-2: 0.997136, top-3: 0.999893
Classifier KNN trained in 5 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.955010, top-1: 0.966674, top-2: 0.995516, top-3: 0.997952
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.983994, top-1: 0.987566, top-2: 0.997363, top-3: 0.997484
Classifier DT trained in 2 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.963065, top-1: 0.972964, top-2: 0.981584, top-3: 0.989012
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.969703, top-1: 0.975066, top-2: 0.978706, top-3: 0.978719
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.936212 | 0.991876 | 0.998782 |            0.901518 |
| KNN | 0.966674 | 0.995516 | 0.997952 |            0.95501  |
| DT  | 0.972964 | 0.981584 | 0.989012 |            0.963065 |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.980339 | 0.997136 | 0.999893 |            0.976009 |
| KNN | 0.987566 | 0.997363 | 0.997484 |            0.983994 |
| DT  | 0.975066 | 0.978706 | 0.978719 |            0.969703 |

python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_Eastgate_24h/Preprocessed/ --test 0.3 --output research_data/Bellevue_Eastgate_24h/Preprocessed/ --preprocessed --min-samples 200 --max-eps 0.16 --mse 0.2
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
Dataset loaded in 214 s
Number of tracks: 32261
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 537752.22it/s]
Shape of feature vectors: (32261, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8]
Number of labeled trajectories after clustering: 32052
Clustering done in 61 s
Reduce labels: 32052it [00:00, 4223955.50it/s]
Clustered exit centroids: [2 0 1 3 2 1 3 1 0]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 22436
Size of testing set: 9616
Feature vectors generated in 0 s
Classifier SVM trained in 150 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.902934, top-1: 0.937164, top-2: 0.991439, top-3: 0.998680
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.975842, top-1: 0.980452, top-2: 0.996754, top-3: 0.999964
^[Classifier KNN trained in 4 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.954572, top-1: 0.966665, top-2: 0.995274, top-3: 0.997860
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.984038, top-1: 0.987301, top-2: 0.997235, top-3: 0.997432
Classifier DT trained in 2 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.962842, top-1: 0.972997, top-2: 0.980755, top-3: 0.988460
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.970021, top-1: 0.975280, top-2: 0.978989, top-3: 0.978989
|     |    Top-1 |    Top-2 |   Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|--------:|--------------------:|
| SVM | 0.937164 | 0.991439 | 0.99868 |            0.902934 |
| KNN | 0.966665 | 0.995274 | 0.99786 |            0.954572 |
| DT  | 0.972997 | 0.980755 | 0.98846 |            0.962842 |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.980452 | 0.996754 | 0.999964 |            0.975842 |
| KNN | 0.987301 | 0.997235 | 0.997432 |            0.984038 |
| DT  | 0.97528  | 0.978989 | 0.978989 |            0.970021 |

 python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_Eastgate_24h/Preprocessed/ --test 0.2 --output research_data/Bellevue_Eastgate_24h/Preprocessed/ --preprocessed --min-samples 200 --max-eps 0.16 --mse 0.2
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
Dataset loaded in 212 s
Number of tracks: 32261
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 557210.50it/s]
Shape of feature vectors: (32261, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8]
Number of labeled trajectories after clustering: 32052
Clustering done in 90 s
Reduce labels: 32052it [00:00, 4627739.48it/s]
Clustered exit centroids: [2 1 0 3 2 0 3 0 1]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 25641
Size of testing set: 6411
Feature vectors generated in 0 s
Classifier SVM trained in 178 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.904016, top-1: 0.938123, top-2: 0.991825, top-3: 0.998691
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.977471, top-1: 0.981378, top-2: 0.997195, top-3: 0.999947
Classifier KNN trained in 3 s
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.955450, top-1: 0.967806, top-2: 0.995752, top-3: 0.998050
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.983872, top-1: 0.987069, top-2: 0.997702, top-3: 0.997969
Classifier DT trained in 3 s
Classifier DT evaluation based on original clusters: balanced accuracy: 0.965888, top-1: 0.974913, top-2: 0.982447, top-3: 0.989767
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.972714, top-1: 0.976222, top-2: 0.980630, top-3: 0.980683
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.938123 | 0.991825 | 0.998691 |            0.904016 |
| KNN | 0.967806 | 0.995752 | 0.99805  |            0.95545  |
| DT  | 0.974913 | 0.982447 | 0.989767 |            0.965888 |
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.981378 | 0.997195 | 0.999947 |            0.977471 |
| KNN | 0.987069 | 0.997702 | 0.997969 |            0.983872 |
| DT  | 0.976222 | 0.98063  | 0.980683 |            0.972714 |

python3 -m sklearnex classification.py --n-jobs 6 exitpoint-metrics --dataset research_data/Bellevue_Eastgate_24h/Preprocessed/ --test 0.05 --output research_data/Bellevue_Eastgate_24h/Preprocessed/ --preprocessed --min-samples 200 --max-eps 0.16 --mse 0.2                                                                                                                                                                                   
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
Dataset loaded in 225 s
Number of tracks: 32261
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 522813.28it/s]
Shape of feature vectors: (32261, 4)
Classes: [-1  0  1  2  3  4  5  6  7  8]
Number of labeled trajectories after clustering: 32052
Clustering done in 73 s
Reduce labels: 32052it [00:00, 4884490.49it/s]
Clustered exit centroids: [2 1 0 3 2 0 3 0 1]
Exit points clusters: [0 1 2 3]
Exit point clustering done in 0 s
Train test split done in 0 s
Size of training set: 30449
Size of testing set: 1603
Feature vectors generated in 0 s
Classifier SVM trained in 265 s
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.908707, top-1: 0.936891, top-2: 0.990785, top-3: 0.998286
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.976210, top-1: 0.979321, top-2: 0.997536, top-3: 0.999893
Killed
(yolov7) pecneb@rigel:/media/pecneb/4d646cbd-cce0-42c4-bdf5-b43cc196e4a1/gitclones/computer_vision_research$ /home/pecneb/anaconda3/envs/yolov7/lib/python3.10/site-packages/joblib/externals/loky/backend/resource_tracker.py:310: UserWarning: resource_tracker: There appear to be 6 leaked semlock objects to clean up at shutdown
  warnings.warn(
/home/pecneb/anaconda3/envs/yolov7/lib/python3.10/site-packages/joblib/externals/loky/backend/resource_tracker.py:310: UserWarning: resource_tracker: There appear to be 1 leaked folder objects to clean up at shutdown
  warnings.warn(