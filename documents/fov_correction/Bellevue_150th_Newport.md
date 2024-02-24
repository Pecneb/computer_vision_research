# Bellevue 150th Newport

## 1

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       1 |                       80.6412 |                              90.6312 |  202.856 |
|   1 | KNN {'n_neighbors': 7}          |       1 |                       95.3102 |                              96.0973 | 0.310949 |
|   2 | DT {'max_depth': None}          |       1 |                         95.36 |                              96.0752 | 0.745506 |

# 1_SG_velocity

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                       90.4732 |                               93.884 |  117.516 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                       94.8416 |                              96.0166 | 0.314387 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                       91.6776 |                              93.4589 |  1.07831 |
