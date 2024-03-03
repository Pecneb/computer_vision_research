# Bellevue 150th Newport

## 1

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       1 |                       80.6412 |                              90.6312 |  202.856 |
|   1 | KNN {'n_neighbors': 7}          |       1 |                       95.3102 |                              96.0973 | 0.310949 |
|   2 | DT {'max_depth': None}          |       1 |                         95.36 |                              96.0752 | 0.745506 |

## 1_SG_velocity m/tick

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                       90.4732 |                               93.884 |  117.516 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                       94.8416 |                              96.0166 | 0.314387 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                       91.6776 |                              93.4589 |  1.07831 |

## m/s

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                       90.6844 |                              94.0452 |  107.968 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                       92.5146 |                              94.6466 | 0.306769 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                        91.541 |                              93.2419 |  1.03549 |

## 7

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       7 |                       86.1363 |                              93.0888 |  246.214 |
|   1 | KNN {'n_neighbors': 7}          |       7 |                       94.4633 |                              95.6485 | 0.341143 |
|   2 | DT {'max_depth': None}          |       7 |                       92.2379 |                              93.5669 | 0.370197 |

## 7_SG_velocity

|     | classifier                      | version        | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------- | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 7_SG_transform |                       89.1126 |                              94.1642 |  334.603 |
|   1 | KNN {'n_neighbors': 7}          | 7_SG_transform |                       91.7958 |                              94.3582 | 0.662708 |
|   2 | DT {'max_depth': None}          | 7_SG_transform |                        89.779 |                              92.2885 |  1.07402 |
