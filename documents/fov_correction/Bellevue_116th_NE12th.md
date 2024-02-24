# Bellevue 116th NE12th

## 1_SG_velocity

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) |  time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | --------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                       84.5146 |                              98.0876 |   2.86077 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                        89.947 |                              98.8317 | 0.0369572 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                       83.5805 |                              92.6723 |  0.113999 |

## 1

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) |  time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | --------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       1 |                       67.4355 |                              83.5434 |   4.07712 |
|   1 | KNN {'n_neighbors': 7}          |       1 |                       90.1315 |                              98.7744 | 0.0365296 |
|   2 | DT {'max_depth': None}          |       1 |                       90.5908 |                              95.4955 | 0.0732767 |
