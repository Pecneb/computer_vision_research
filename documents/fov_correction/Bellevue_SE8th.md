# Bellevue SE8th

## 1

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       1 |                       76.6606 |                              91.1231 |  82.2881 |
|   1 | KNN {'n_neighbors': 7}          |       1 |                       93.9919 |                              97.9703 | 0.339768 |
|   2 | DT {'max_depth': None}          |       1 |                       91.8356 |                              95.1947 | 0.771302 |

# 1_SG_velocity

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                       85.6732 |                              97.2815 |  61.1282 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                       94.0664 |                              98.0486 | 0.347544 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                       89.7144 |                              94.3875 |  1.27647 |
