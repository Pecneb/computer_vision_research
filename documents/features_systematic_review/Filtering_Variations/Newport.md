# Bellevue 150th Newport results with varying filtering methods

## Raw dataset

|     | classifier                                                                                                                                                                      | version | split_0 (percent) | split_1 (percent) | split_2 (percent) | split_3 (percent) | split_4 (percent) | mean (percent) | std (percent) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: | -------------: | ------------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |           75.5074 |           83.1229 |           80.9263 |            81.555 |            81.134 |        80.4491 |       2.58804 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |           75.8123 |            84.139 |           83.8403 |           82.3698 |           83.5839 |         81.949 |       3.12671 |

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       82.5831 |                              86.2013 | 0.830225 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                        83.237 |                              88.7931 |   3089.9 |

Total time: 11460.226493141003 seconds

## enter exit distance filter

|     | classifier                                                                                                                                                                      | version | split_0 (percent) | split_1 (percent) | split_2 (percent) | split_3 (percent) | split_4 (percent) | mean (percent) | std (percent) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: | -------------: | ------------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |           93.1331 |           92.8269 |           92.8428 |           92.3813 |           92.1445 |        92.6657 |      0.354496 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |           93.5671 |             91.65 |           93.2348 |           91.5144 |           93.4055 |        92.6744 |      0.898949 |

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       93.8013 |                              95.1747 | 0.492266 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       94.7871 |                              95.6057 |  2652.29 |

Total time: 7675.213463208987 seconds

## enter exit distance plus edge distance

|     | classifier                                                                                                                                                                      | version | split_0 (percent) | split_1 (percent) | split_2 (percent) | split_3 (percent) | split_4 (percent) | mean (percent) | std (percent) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: | -------------: | ------------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |           93.1331 |           92.8269 |           92.8428 |           92.3813 |           92.1445 |        92.6657 |      0.354496 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |           93.7217 |            92.457 |           92.6355 |           92.2553 |           92.1726 |        92.6484 |      0.560295 |

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       93.8013 |                              95.1747 | 0.467682 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                         94.34 |                              96.2566 |  2045.58 |

Total time: 6702.0577776860155 seconds

## enter exit distance plus edge distance and fov correction

| classifier | version | split_0 (percent) | split_1 (percent) | split_2 (percent) | split_3 (percent) | split_4 (percent) | mean (percent) | std (percent) |
| ---------- | ------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | -------------- | ------------- |

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       93.2508 |                              94.6041 |  1.03688 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                         95.29 |                              95.9987 |  1181.83 |

Total time: 1606.063691834017 seconds
