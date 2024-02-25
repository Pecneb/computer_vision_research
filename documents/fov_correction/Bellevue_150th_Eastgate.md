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
