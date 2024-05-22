# Bellevue Eastgate

## Raw

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       95.1095 |                              96.0628 |  1.05549 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       84.4975 |                              98.1885 |  431.164 |

Total time: 1440.767215167998 seconds

Number of tracks: 92387  
Number of detections: 11591231  
Average detections per track: 125.46387478757833  
Max detections per track: 3000 Min detections per track: 3  
Standard deviation: 286.9457215252935 Max distance: 1.6493583589578293  
Min distance: 0.0

X_train: (10312,), Y_train: (10312,), Y_pooled_train: (10312,), X_test: (2579,), Y_test: (2579,), Y_pooled_test: (2579,)

## distance based filter

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       86.8041 |                              93.3819 |  1.27446 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       87.2269 |                              93.5863 |   1335.9 |

Total time: 1988.4087740399991 seconds

Number of tracks: 92387
Number of detections: 11591231
Average detections per track: 125.46387478757833
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 286.9457215252935
Max distance: 1.6493583589578293
Min distance: 0.0
Dataset statistics after preprocessing:
Number of tracks: 32261
Number of detections: 7119070
Average detections per track: 220.67108893090727
Max detections per track: 3000
Min detections per track: 5
Standard deviation: 339.87065183273194
Max distance: 1.6493583589578293
Min distance: 0.40000779272549614

X_train: (20946,), Y_train: (20946,), Y_pooled_train: (20946,), X_test: (5237,), Y_test: (5237,), Y_pooled_test: (5237,)

## distance based and edge distance based filter

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       86.8041 |                              93.3819 |  1.15648 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                        86.393 |                              93.3826 |  1653.69 |

Total time: 2264.365424639 seconds

Dataset statistics before preprocessing:
Number of tracks: 92387
Number of detections: 11591231
Average detections per track: 125.46387478757833
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 286.9457215252935
Max distance: 1.6493583589578293
Min distance: 0.0

Dataset statistics after preprocessing:
Number of tracks: 32261
Number of detections: 7119070
Average detections per track: 220.67108893090727
Max detections per track: 3000
Min detections per track: 5
Standard deviation: 339.87065183273194
Max distance: 1.6493583589578293
Min distance: 0.40000779272549614

X_train: (20946,), Y_train: (20946,), Y_pooled_train: (20946,), X_test: (5237,), Y_test: (5237,), Y_pooled_test: (5237,)


## distance based and edge distance based filter and fov correction

Dataset statistics before preprocessing:                                                                                                                                                                                                                      
Number of tracks: 92387                                                                                                                                                                                                                                       
Number of detections: 11591231                                                                                                                                                                                                                                
Average detections per track: 125.46387478757833                                                                                                                                                                                                              
Max detections per track: 3000                                                                                                                                                                                                                                
Min detections per track: 3                                                                                                                                                                                                                                   
Standard deviation: 286.9457215252935                                                                                                                                                                                                                         
Max distance: 1.6493583589578293                                                                                                                                                                                                                              
Min distance: 0.0                                                                                                                                                                                                                                             
Filter out edge detections.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 186487.63it/s]
Dataset statistics after preprocessing:                                                                                                                                                                                                                       
Number of tracks: 32261                                                                                                                                                                                                                                       
Number of detections: 7119070                                                                                                                                                                                                                                 
Average detections per track: 220.67108893090727                                                                                                                                                                                                              
Max detections per track: 3000                                                                                                                                                                                                                                
Min detections per track: 5                                                                                                                                                                                                                                   
Standard deviation: 339.87065183273194                                                                                                                                                                                                                        
Max distance: 1.6493583589578293                                                                                                                                                                                                                              
Min distance: 0.40000779272549614                                                                                                                                                                                                                             
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 377037.76it/s]
Classes: [-1  0  1  2  3  4  5  6  7  8]                                                                                                                                                                                                                      
Reduce labels: 26183it [00:00, 4551913.36it/s]                                                                                                                                                                                                                
Pooled classes: [3 0 2 3 2 0 1 3 1]                                                                                                                                                                                                                           
X_train: (20946,), Y_train: (20946,), Y_pooled_train: (20946,), X_test: (5237,), Y_test: (5237,), Y_pooled_test: (5237,)                                                                                                                                      
Features for classification.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20946/20946 [00:34<00:00, 608.17it/s]
Features for classification.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5237/5237 [00:08<00:00, 617.86it/s][Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done   9 out of   9 | elapsed:    1.5s finished                                                                                                                                                                                         Balanced Test score: 0.8691028464875724
Balanced Pooled Test score: 0.9335426534472526                                                                                                                                                                                                                [Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done   9 out of   9 | elapsed: 11.6min finished
Balanced Test score: 0.8812184541654763
Balanced Pooled Test score: 0.9379539848025512
| classifier   | version   | split_0 (percent)   | split_1 (percent)   | split_2 (percent)   | split_3 (percent)   | split_4 (percent)   | mean (percent)   | std (percent)   |
|--------------|-----------|---------------------|---------------------|---------------------|---------------------|---------------------|------------------|-----------------|
|    | classifier                                                                                                                                                                      | version   |   balanced_test_score (percent) |   balanced_pooled_test_
score (percent) |   time (s) |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|--------------------------------:|------------------------
---------------:|-----------:|
|  0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs    |                         86.9103 |                        
        93.3543 |    1.87473 |
|  1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs    |                         88.1218 |                        
        93.7954 |  698.798   |
Total time: 1370.401302798 seconds