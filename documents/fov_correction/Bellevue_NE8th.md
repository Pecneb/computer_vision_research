# Bellevue NE8th

## 1

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       1 |                       77.9356 |                              96.3054 |  611.568 |
|   1 | KNN {'n_neighbors': 7}          |       1 |                       92.5139 |                              98.7744 |  1.61854 |
|   2 | DT {'max_depth': None}          |       1 |                       93.1055 |                              97.2613 |  1.65309 |

## 1_SG_velocity m/tick

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       1 |                       86.4484 |                              97.8886 |  348.035 |
|   1 | KNN {'n_neighbors': 7}          |       1 |                       92.4827 |                              98.7733 |  1.10096 |
|   2 | DT {'max_depth': None}          |       1 |                       93.0317 |                              97.2565 |  2.33479 |

### m/s

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                       87.8581 |                              98.0174 |  231.323 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                       90.6668 |                              98.5414 | 0.576066 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                       89.6207 |                              95.9668 |  3.05878 |

# 7

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       7 |                       83.3526 |                              95.9027 |  2768.71 |
|   1 | KNN {'n_neighbors': 7}          |       7 |                       89.6175 |                              96.9374 |  2.40039 |
|   2 | DT {'max_depth': None}          |       7 |                       88.4289 |                              95.0383 |  1.80842 |

# 7_SG_velocity

|     | classifier                      | version        | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------- | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 7_SG_transform |                        82.034 |                               95.281 |   2271.8 |
|   1 | KNN {'n_neighbors': 7}          | 7_SG_transform |                        87.059 |                              96.6376 |   2.8345 |
|   2 | DT {'max_depth': None}          | 7_SG_transform |                       86.1302 |                              94.4612 |   4.6848 |
