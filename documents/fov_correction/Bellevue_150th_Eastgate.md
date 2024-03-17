# Bellevue 150th Eastgate

## 1

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       1 |                       77.4469 |                              91.6208 |  274.487 |
|   1 | KNN {'n_neighbors': 7}          |       1 |                        94.365 |                              98.6209 | 0.521371 |
|   2 | DT {'max_depth': None}          |       1 |                       95.3035 |                              97.7078 |  0.95619 |

## 1_SG_velocity m/tick

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                       87.1069 |                              95.4485 |  186.594 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                       93.2472 |                              98.3371 | 0.421946 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                       90.6054 |                              96.1247 |  2.07602 |

## m/s

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                       89.2398 |                              95.8189 |   189.41 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                       92.7478 |                              97.7956 | 0.410096 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                       89.9285 |                               96.068 |  2.12414 |

## 7

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       7 |                        82.854 |                              90.4688 |  400.029 |
|   1 | KNN {'n_neighbors': 7}          |       7 |                       88.0899 |                               93.967 | 0.863047 |
|   2 | DT {'max_depth': None}          |       7 |                       85.8853 |                              92.7179 |  0.69158 |

## 7_SG_velocity

|     | classifier                      | version        | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------- | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 7_SG_transform |                       82.8618 |                              91.6412 |  568.414 |
|   1 | KNN {'n_neighbors': 7}          | 7_SG_transform |                       85.4204 |                              93.3562 |  1.00403 |
|   2 | DT {'max_depth': None}          | 7_SG_transform |                       84.3562 |                              92.3589 |  1.85895 |

# 11

|     | classifier                        | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :-------------------------------- | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 100, 'max_iter': 60000} |      11 |                       83.9624 |                               93.668 |  455.508 |
|   1 | KNN {'n_neighbors': 7}            |      11 |                       82.5959 |                              93.1016 |   1.0854 |
|   2 | DT {'max_depth': None}            |      11 |                       82.3386 |                              91.9145 | 0.903618 |