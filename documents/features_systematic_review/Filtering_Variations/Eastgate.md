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
Classes: [-1 0 1 2 3 4 5 6 7 8]  
Reduce labels: 26183it [00:00, 4551913.36it/s]  
Pooled classes: [3 0 2 3 2 0 1 3 1]  
X\*train: (20946,), Y_train: (20946,), Y_pooled_train: (20946,), X_test: (5237,), Y_test: (5237,), Y_pooled_test: (5237,)  
Features for classification.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20946/20946 [00:34<00:00, 608.17it/s]
Features for classification.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5237/5237 [00:08<00:00, 617.86it/s][Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.

| classifier | version | balanced_test_score (percent) | balanced_pooled_test\*\*
core (percent) | time (s) |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|--------------------------------:|---------------------------------------:|-----------:|
| 0 | KNN {'n_neighbors': 7} | ReVeRs | 86.9103 |  
 93.3543 | 1.87473 |
| 1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs | 88.1218 |  
 93.7954 | 698.798 |
Total time: 1370.401302798 seconds

# Second Run

## Raw

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 100, 'max_iter': 60000}                                                                                                                                               | ReVeRs  |                       94.1758 |                              97.8643 |  46.2072 |
|   1 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       95.1095 |                              96.0628 | 0.256253 |
|   2 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                        82.423 |                               94.546 |  433.176 |

**Total time:** `1387.9501615910012 seconds`

### Dataset statistics before preprocessing:

Number of tracks: 92387
Number of detections: 11591231
Average detections per track: 125.46387478757833
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 286.9457215252935
Max distance: 1.6493583589578293
Min distance: 0.0

### Dataset statistics after preprocessing:

Number of tracks: 92387
Number of detections: 11591231
Average detections per track: 125.46387478757833
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 286.9457215252935
Max distance: 1.6493583589578293
Min distance: 0.0

### Clustering

Classes: [-1 0 1 2 3 4 5 6 7 8]
Pooled classes: [3 0 1 3 0 0 2 1 1]

### Feature Vector Count

X_train: (10312,), Y_train: (10312,), Y_pooled_train: (10312,), X_test: (2579,), Y_test: (2579,), Y_pooled_test: (2579,)

## Enter Exit Distance Based Filter

### Dataset statistics before preprocessing:

Number of tracks: 92387
Number of detections: 11591231
Average detections per track: 125.46387478757833
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 286.9457215252935
Max distance: 1.6493583589578293
Min distance: 0.0

### Dataset statistics after preprocessing:

Number of tracks: 32261
Number of detections: 7119070
Average detections per track: 220.67108893090727
Max detections per track: 3000
Min detections per track: 5
Standard deviation: 339.87065183273194
Max distance: 1.6493583589578293
Min distance: 0.40000779272549614

Classes: [-1 0 1 2 3 4 5 6 7 8]
Pooled classes: [0 3 1 0 1 3 2 0 2]

X_train: (20946,), Y_train: (20946,), Y_pooled_train: (20946,), X_test: (5237,), Y_test: (5237,), Y_pooled_test: (5237,)

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 100, 'max_iter': 60000}                                                                                                                                               | ReVeRs  |                       86.8203 |                              93.2267 |  342.422 |
|   1 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       86.8041 |                              93.3819 | 0.427944 |
|   2 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       86.6941 |                               93.214 |   1299.7 |

Total time: 2334.7888125070003 seconds

## Enter Exit Distance Based Filter and Edge Distance Filter

### Dataset statistics before preprocessing:

Number of tracks: 92387
Number of detections: 11591231
Average detections per track: 125.46387478757833
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 286.9457215252935
Max distance: 1.6493583589578293
Min distance: 0.0

### Dataset statistics after preprocessing:

Number of tracks: 32261
Number of detections: 7119070
Average detections per track: 220.67108893090727
Max detections per track: 3000
Min detections per track: 5
Standard deviation: 339.87065183273194
Max distance: 1.6493583589578293
Min distance: 0.40000779272549614

Classes: [-1 0 1 2 3 4 5 6 7 8]
Reduce labels: 26183it [00:00, 4729316.64it/s]
Pooled classes: [0 3 1 0 1 3 2 0 2]
X_train: (20946,), Y_train: (20946,), Y_pooled_train: (20946,), X_test: (5237,), Y_test: (5237,), Y_pooled_test: (5237,)
Features for classification.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20946/20946 [00:33<00:00, 627.71it/s]
Features for classification.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5237/5237 [00:08<00:00, 638.49it/s]
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.8/site-packages/sklearn/svm/\_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=60000). Consider pre-processing your data with StandardScaler or MinMaxScaler.
warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.8/site-packages/sklearn/svm/\_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=60000). Consider pre-processing your data with StandardScaler or MinMaxScaler.
warnings.warn(
[Parallel(n_jobs=4)]: Done 9 out of 9 | elapsed: 6.0min finished
Balanced Test score: 0.8682034463862677
Balanced Pooled Test score: 0.9322670782182265
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done 9 out of 9 | elapsed: 0.3s finished
Balanced Test score: 0.8680411990979633
Balanced Pooled Test score: 0.9338189846962383
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done 9 out of 9 | elapsed: 20.6min finished
Balanced Test score: 0.8591159310262645
Balanced Pooled Test score: 0.9283875667832256

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 100, 'max_iter': 60000}                                                                                                                                               | ReVeRs  |                       86.8203 |                              93.2267 |  358.326 |
|   1 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       86.8041 |                              93.3819 | 0.444236 |
|   2 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       85.9116 |                              92.8388 |  1234.37 |

**Total time:** `2295.813094432 seconds`

## Enter Exit Distance Based Filter and Edge Distance Filter and Geometric Transform

Loading datasets: 23it [05:52, 15.31s/it]
Dataset statistics before preprocessing:
Number of tracks: 92387
Number of detections: 11591231
Average detections per track: 125.46387478757833
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 286.9457215252935
Max distance: 1.6493583589578293
Min distance: 0.0
Filter out edge detections.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 214433.11it/s]
Dataset statistics after preprocessing:
Number of tracks: 32261
Number of detections: 7119070
Average detections per track: 220.67108893090727
Max detections per track: 3000
Min detections per track: 5
Standard deviation: 339.87065183273194
Max distance: 1.6493583589578293
Min distance: 0.40000779272549614
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 437040.16it/s]
Classes: [-1 0 1 2 3 4 5 6 7 8]
Reduce labels: 26183it [00:00, 4686730.18it/s]
Pooled classes: [1 0 2 1 2 0 3 1 3]
X_train: (20946,), Y_train: (20946,), Y_pooled_train: (20946,), X_test: (5237,), Y_test: (5237,), Y_pooled_test: (5237,)
Features for classification.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20946/20946 [00:33<00:00, 623.28it/s]
Features for classification.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5237/5237 [00:08<00:00, 639.73it/s]
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.8/site-packages/sklearn/svm/\_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=60000). Consider pre-processing your data with StandardScaler or MinMaxScaler.
warnings.warn(
/home/pecneb/miniconda3/envs/computer-vision/lib/python3.8/site-packages/sklearn/svm/\_base.py:297: ConvergenceWarning: Solver terminated early (max_iter=60000). Consider pre-processing your data with StandardScaler or MinMaxScaler.
warnings.warn(
[Parallel(n_jobs=4)]: Done 9 out of 9 | elapsed: 4.7min finished
Balanced Test score: 0.8827878599838487
Balanced Pooled Test score: 0.9402888993430567
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done 9 out of 9 | elapsed: 0.3s finished
Balanced Test score: 0.8691028464875724
Balanced Pooled Test score: 0.9335426534472526
[Parallel(n_jobs=4)]: Using backend LokyBackend with 4 concurrent workers.
[Parallel(n_jobs=4)]: Done 9 out of 9 | elapsed: 9.9min finished
Balanced Test score: 0.8811497309432959
Balanced Pooled Test score: 0.9374763864802629
| | classifier | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|--------------------------------:|---------------------------------------:|-----------:|
| 0 | SVM {'C': 100, 'max_iter': 60000} | ReVeRs | 88.2788 | 94.0289 | 282.382 |
| 1 | KNN {'n_neighbors': 7} | ReVeRs | 86.9103 | 93.3543 | 0.420006 |
| 2 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs | 88.115 | 93.7476 | 596.461 |
Total time: 1636.0402202219993 seconds
