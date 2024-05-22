# Bellevue NE12th

## raw

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       79.6026 |                              91.7399 | 0.574009 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                        71.797 |                               92.271 |  68.1229 |

**Total time:** `243.87775846799923 seconds`

### Dataset statistics before preprocessing:

Number of tracks: 19316
Number of detections: 1836736
Average detections per track: 95.08883826879271
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 237.2101952310517
Max distance: 1.6333056375615485
Min distance: 0.0

### Dataset statistics after preprocessing:

Number of tracks: 19316
Number of detections: 1836736
Average detections per track: 95.08883826879271
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 237.2101952310517
Max distance: 1.6333056375615485
Min distance: 0.0

### Clustering

Classes: [-1 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22]
Pooled classes: [4 4 4 0 0 0 5 1 1 3 3 3 0 0 2 3 2 1 2 3 0 0 1]

### Feature Vector count

X_train: (7216,), Y_train: (7216,), Y_pooled_train: (7216,), X_test: (1805,), Y_test: (1805,), Y_pooled_test: (1805,)

## enter exit distance filter

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       89.4363 |                              98.0517 | 0.438576 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       89.2277 |                              98.7089 |  31.7729 |

**Total time:** `111.55541252799958 seconds`

### Dataset statistics before preprocessing:

Number of tracks: 19316
Number of detections: 1836736
Average detections per track: 95.08883826879271
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 237.2101952310517
Max distance: 1.6333056375615485
Min distance: 0.0

### Dataset statistics after preprocessing:

Number of tracks: 3836
Number of detections: 1199438
Average detections per track: 312.6793534932221
Max detections per track: 3000
Min detections per track: 10
Standard deviation: 391.1995321219451
Max distance: 1.6333056375615485
Min distance: 0.40044163420976847

### Clustering

Classes: [-1 0 1 2 3 4 5 6 7 8 9]
Pooled classes: [0 1 0 2 3 2 1 0 0 3]

### Feature Vector count

X_train: (1783,), Y_train: (1783,), Y_pooled_train: (1783,), X_test: (446,), Y_test: (446,), Y_pooled_test: (446,)

## enter exit distance and edge distance based filter

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       89.4363 |                              98.0517 | 0.443278 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       87.0025 |                              98.6878 |  22.2355 |

### Dataset statistics before preprocessing:

Number of tracks: 19316
Number of detections: 1836736
Average detections per track: 95.08883826879271
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 237.2101952310517
Max distance: 1.6333056375615485
Min distance: 0.0

### Dataset statistics after preprocessing:

Number of tracks: 3836
Number of detections: 1199438
Average detections per track: 312.6793534932221
Max detections per track: 3000
Min detections per track: 10
Standard deviation: 391.1995321219451
Max distance: 1.6333056375615485
Min distance: 0.40044163420976847

### Clustering

Classes: [-1 0 1 2 3 4 5 6 7 8 9]
Pooled classes: [0 1 0 2 3 2 1 0 0 3]

### Feature Vector count

X_train: (1783,), Y_train: (1783,), Y_pooled_train: (1783,), X_test: (446,), Y_test: (446,), Y_pooled_test: (446,)

**Total time:** `103.89572978100114 seconds`

## enter exit distance, edge distance based and fov correction

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       89.8548 |                              98.2834 | 0.474073 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       89.1317 |                              98.6238 |  39.8726 |

**Total time:** `136.42280498500077 seconds`

### Dataset statistics before preprocessing:

Number of tracks: 19316
Number of detections: 1836736
Average detections per track: 95.08883826879271
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 237.2101952310517
Max distance: 1.6333056375615485
Min distance: 0.0

### Dataset statistics after preprocessing:

Number of tracks: 3836
Number of detections: 1199438
Average detections per track: 312.6793534932221
Max detections per track: 3000
Min detections per track: 10
Standard deviation: 391.1995321219451
Max distance: 1.6333056375615485
Min distance: 0.40044163420976847

### Clustering

Classes: [-1 0 1 2 3 4 5 6 7 8 9]
Pooled classes: [1 0 1 2 3 2 0 1 1 3]

### Feature Vector count

X_train: (1783,), Y_train: (1783,), Y_pooled_train: (1783,), X_test: (446,), Y_test: (446,), Y_pooled_test: (446,)
