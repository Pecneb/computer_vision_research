|     | classifier                                                                                                                                                                      | version | split_0 (percent) | split_1 (percent) | split_2 (percent) | split_3 (percent) | split_4 (percent) | mean (percent) | std (percent) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: | -------------: | ------------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |           86.4031 |           92.8804 |           86.2799 |           90.2492 |           93.8141 |        89.9253 |       3.15134 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |           73.4059 |           73.0919 |           70.0044 |           71.8986 |           73.4632 |        72.3728 |       1.31208 |

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       84.8038 |                              98.5431 |  1.24004 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       66.8405 |                              98.8859 |  2310.57 |

Total time: 8623.355388319003 seconds

## enter exit distance based filter

|     | classifier                                                                                                                                                                      | version | split_0 (percent) | split_1 (percent) | split_2 (percent) | split_3 (percent) | split_4 (percent) | mean (percent) | std (percent) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------: | ----------------: | ----------------: | ----------------: | ----------------: | -------------: | ------------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |           90.3018 |           89.9482 |            90.087 |           90.1055 |            90.232 |        90.1349 |      0.122717 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |           89.2027 |           89.2005 |           88.8419 |           88.7379 |           88.5822 |         88.913 |      0.249716 |

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       90.1997 |                              98.1886 | 0.826068 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       88.9278 |                              98.7406 |  2266.66 |

Total time: 8481.641022962 seconds

## enter exit distance and edge distance based filter

| classifier | version | split_0 (percent) | split_1 (percent) | split_2 (percent) | split_3 (percent) | split_4 (percent) | mean (percent) | std (percent) |
| ---------- | ------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | -------------- | ------------- |

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                       90.1997 |                              98.1886 |  1.57786 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       88.7154 |                              98.7268 |  2845.46 |

Total time: 3310.446339600996 seconds

## enter exit distance and edge distance based filter and fov correction

| classifier | version | split_0 (percent) | split_1 (percent) | split_2 (percent) | split_3 (percent) | split_4 (percent) | mean (percent) | std (percent) |
| ---------- | ------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | -------------- | ------------- |

|     | classifier                                                                                                                                                                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs  |                        90.266 |                              98.1244 |  1.68188 |
|   1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs  |                       89.6971 |                              98.6984 |  1099.21 |

Total time: 1668.9747433300072 seconds
