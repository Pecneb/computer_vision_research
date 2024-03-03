# Bellevue SE8th

## 1

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       1 |                       76.6606 |                              91.1231 |  82.2881 |
|   1 | KNN {'n_neighbors': 7}          |       1 |                       93.9919 |                              97.9703 | 0.339768 |
|   2 | DT {'max_depth': None}          |       1 |                       91.8356 |                              95.1947 | 0.771302 |

## 1_SG_velocity m/tick

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                       85.6732 |                              97.2815 |  61.1282 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                       94.0664 |                              98.0486 | 0.347544 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                       89.7144 |                              94.3875 |  1.27647 |

### m/s

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                       87.3579 |                              97.2084 |  60.8504 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                       92.8677 |                              97.8838 | 0.361149 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                       89.2013 |                              94.7971 |   1.3093 |

# 7

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       7 |                       77.4167 |                              91.7891 |  318.671 |
|   1 | KNN {'n_neighbors': 7}          |       7 |                        87.541 |                              97.4455 | 0.409755 |
|   2 | DT {'max_depth': None}          |       7 |                       84.3732 |                              94.4111 | 0.537709 |

# 7_SG_velocity

|     | classifier                      | version        | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------- | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 7_SG_transform |                       80.9155 |                              97.6757 |  229.525 |
|   1 | KNN {'n_neighbors': 7}          | 7_SG_transform |                       84.6023 |                              97.0226 | 0.389651 |
|   2 | DT {'max_depth': None}          | 7_SG_transform |                       80.8274 |                              94.1086 |  1.36229 |
