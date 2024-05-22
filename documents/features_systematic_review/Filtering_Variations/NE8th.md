# Bellevue NE8th

## Raw

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       92.1028 |                              97.7455 |  3.05924 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                         77.26 |                              95.2558 |  3376.91 |

**Total time:** `4429 seconds`

### Dataset statistics before preprocessing:

Number of tracks: 101703
Number of detections: 18841575
Average detections per track: 185.26075926963807
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 426.9578886470609
Max distance: 1.6928937413722394
Min distance: 0.0

### Dataset statistics after preprocessing:

Number of tracks: 101703
Number of detections: 18841575
Average detections per track: 185.26075926963807
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 426.9578886470609
Max distance: 1.6928937413722394
Min distance: 0.0

### Clustering

Classes: [-1 0 1 2 3 4 5 6 7 8 9 10 11 12]
Pooled classes: [4 4 3 1 2 1 1 0 2 1 2 0 3]

### Feature Vector count

X_train: (27428,), Y_train: (27428,), Y_pooled_train: (27428,), X_test: (6857,), Y_test: (6857,), Y_pooled_test: (6857,)

## enter exit distance based filter

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       87.9025 |                              96.4962 |  2.73953 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       87.2556 |                              96.9212 |  2884.03 |

**Total time:** `3792.7261847389964 seconds`

### Dataset statistics before preprocessing:

Number of tracks: 101703
Number of detections: 18841575
Average detections per track: 185.26075926963807
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 426.9578886470609
Max distance: 1.6928937413722394
Min distance: 0.0

### Dataset statistics after preprocessing:

Number of tracks: 33580
Number of detections: 13360036
Average detections per track: 397.8569386539607
Max detections per track: 3000
Min detections per track: 4
Standard deviation: 606.2902062518752
Max distance: 1.6928937413722394
Min distance: 0.4000204684713268

### Clustering

Classes: [-1 0 1 2 3 4 5 6 7 8 9 10]
Pooled classes: [3 1 1 0 3 2 1 3 0 2 2]

### Feature vector count

X_train: (25620,), Y_train: (25620,), Y_pooled_train: (25620,), X_test: (6405,), Y_test: (6405,), Y_pooled_test: (6405,)

## enter exit distance based filter and edge distance

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       87.9025 |                              96.4962 |   2.9439 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                        86.233 |                              96.8141 |  3203.77 |

### Dataset statistics before preprocessing:

Number of tracks: 101703
Number of detections: 18841575
Average detections per track: 185.26075926963807
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 426.9578886470609
Max distance: 1.6928937413722394
Min distance: 0.0

### Dataset statistics after preprocessing:

Number of tracks: 33580
Number of detections: 13360036
Average detections per track: 397.8569386539607
Max detections per track: 3000
Min detections per track: 4
Standard deviation: 606.2902062518752
Max distance: 1.6928937413722394
Min distance: 0.4000204684713268

### Clustering

Classes: [-1 0 1 2 3 4 5 6 7 8 9 10]
Pooled classes: [1 0 0 3 1 2 0 1 3 2 2]

### Feature vector count

X_train: (25620,), Y_train: (25620,), Y_pooled_train: (25620,), X_test: (6405,), Y_test: (6405,), Y_pooled_test: (6405,)

## enter exit distance based filter and edge distance and fov correction

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       88.0288 |                              96.5508 |  3.59061 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       85.9987 |                              96.6011 |  1965.89 |

### Dataset statistics before preprocessing:

Number of tracks: 101703
Number of detections: 18841575
Average detections per track: 185.26075926963807
Max detections per track: 3000
Min detections per track: 3
Standard deviation: 426.9578886470609
Max distance: 1.6928937413722394
Min distance: 0.0

### Dataset statistics after preprocessing:

Number of tracks: 33580
Number of detections: 13360036
Average detections per track: 397.8569386539607
Max detections per track: 3000
Min detections per track: 4
Standard deviation: 606.2902062518752
Max distance: 1.6928937413722394
Min distance: 0.4000204684713268

### Clustring

Classes: [-1 0 1 2 3 4 5 6 7 8 9 10]
Pooled classes: [2 1 1 3 2 0 1 2 3 0 0]

### Feature vector count

X_train: (25620,), Y_train: (25620,), Y_pooled_train: (25620,), X_test: (6405,), Y_test: (6405,), Y_pooled_test: (6405,)
