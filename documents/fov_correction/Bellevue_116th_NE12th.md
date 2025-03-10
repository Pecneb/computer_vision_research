# Bellevue 116th NE12th

## 1_SG_velocity m/tick

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) |  time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | --------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                       84.5146 |                              98.0876 |   2.86077 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                        89.947 |                              98.8317 | 0.0369572 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                       83.5805 |                              92.6723 |  0.113999 |

### m/s

|     | classifier                      | version       | balanced_test_score (percent) | balanced_pooled_test_score (percent) |  time (s) |
| --: | :------------------------------ | :------------ | ----------------------------: | -----------------------------------: | --------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 1_SG_velocity |                        86.668 |                              97.6251 |   2.60745 |
|   1 | KNN {'n_neighbors': 7}          | 1_SG_velocity |                       89.8861 |                              98.3165 | 0.0362377 |
|   2 | DT {'max_depth': None}          | 1_SG_velocity |                       83.2586 |                              93.2288 |  0.172566 |

## 1

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) |  time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | --------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       1 |                       67.4355 |                              83.5434 |   4.07712 |
|   1 | KNN {'n_neighbors': 7}          |       1 |                       90.1315 |                              98.7744 | 0.0365296 |
|   2 | DT {'max_depth': None}          |       1 |                       90.5908 |                              95.4955 | 0.0732767 |

## 7

|     | classifier                      | version | balanced_test_score (percent) | balanced_pooled_test_score (percent) |  time (s) |
| --: | :------------------------------ | ------: | ----------------------------: | -----------------------------------: | --------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} |       7 |                       83.9847 |                              95.6107 |   10.3412 |
|   1 | KNN {'n_neighbors': 7}          |       7 |                       91.1269 |                              98.4608 | 0.0550471 |
|   2 | DT {'max_depth': None}          |       7 |                       87.2747 |                              95.2492 | 0.0746678 |

## 7_SG_velocity

|     | classifier                      | version        | balanced_test_score (percent) | balanced_pooled_test_score (percent) | time (s) |
| --: | :------------------------------ | :------------- | ----------------------------: | -----------------------------------: | -------: |
|   0 | SVM {'C': 1, 'max_iter': 30000} | 7_SG_transform |                       85.3939 |                              97.8557 |  6.75061 |
|   1 | KNN {'n_neighbors': 7}          | 7_SG_transform |                       87.6115 |                              98.1038 | 0.044303 |
|   2 | DT {'max_depth': None}          | 7_SG_transform |                       83.9477 |                              95.2886 | 0.157845 |

# 11 (x,y,vx,vy)

|---:|:----------------------------------|----------:|--------------------------------:|---------------------------------------:|-----------:|
|  0 | SVM {'C': 100, 'max_iter': 30000} |        11 |                         83.7852 |                                98.6422 |  6.48715   |
|  1 | KNN {'n_neighbors': 7}            |        11 |                         83.4388 |                                97.2381 |  0.0318463 |
|  2 | DT {'max_depth': None}            |        11 |                         83.2317 |                                96.0264 |  0.0929321 |