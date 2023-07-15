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
