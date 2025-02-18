| classifier   | version   | split_0 (percent)   | split_1 (percent)   | split_2 (percent)   | split_3 (percent)   | split_4 (percent)   | mean (percent)   | std (percent)   |
|--------------|-----------|---------------------|---------------------|---------------------|---------------------|---------------------|------------------|-----------------|

|    | classifier                        |   version |   balanced_test_score (percent) |   balanced_pooled_test_score (percent) |   time (s) |
|---:|:----------------------------------|----------:|--------------------------------:|---------------------------------------:|-----------:|
|  0 | SVM {'C': 0.1, 'max_iter': 5000}  |         7 |                         68.0217 |                                76.8913 | 153.49     |
|  1 | SVM {'C': 0.1, 'max_iter': 15000} |         7 |                         80.3326 |                                91.0651 | 338.354    |
|  2 | SVM {'C': 0.1, 'max_iter': 30000} |         7 |                         80.3326 |                                91.0651 | 324.126    |
|  3 | SVM {'C': 1, 'max_iter': 5000}    |         7 |                         82.0432 |                                88.8809 | 175.287    |
|  4 | SVM {'C': 1, 'max_iter': 15000}   |         7 |                         86.1363 |                                93.0888 | 267.159    |
|  5 | SVM {'C': 1, 'max_iter': 30000}   |         7 |                         86.1363 |                                93.0888 | 263.677    |
|  6 | SVM {'C': 10, 'max_iter': 5000}   |         7 |                         92.9815 |                                95.4776 | 130.81     |
|  7 | SVM {'C': 10, 'max_iter': 15000}  |         7 |                         92.9362 |                                95.5581 | 164.795    |
|  8 | SVM {'C': 10, 'max_iter': 30000}  |         7 |                         92.9362 |                                95.5581 | 139.02     |
|  9 | KNN {'n_neighbors': 3}            |         7 |                         94.1888 |                                95.285  |   0.337926 |
| 10 | KNN {'n_neighbors': 5}            |         7 |                         94.4989 |                                95.6145 |   0.343744 |
| 11 | KNN {'n_neighbors': 10}           |         7 |                         94.4719 |                                95.4222 |   0.343108 |
| 12 | DT {'max_depth': None}            |         7 |                         91.9723 |                                93.3718 |   0.513111 |
| 13 | DT {'max_depth': 5}               |         7 |                         92.6564 |                                95.0171 |   0.29985  |
| 14 | DT {'max_depth': 10}              |         7 |                         93.8274 |                                95.4478 |   0.412873 |

| classifier   | version   | split_0 (percent)   | split_1 (percent)   | split_2 (percent)   | split_3 (percent)   | split_4 (percent)   | mean (percent)   | std (percent)   |
|--------------|-----------|---------------------|---------------------|---------------------|---------------------|---------------------|------------------|-----------------|
|    | classifier                         |   version |   balanced_test_score (percent) |   balanced_pooled_test_score (percent) |   time (s) |
|---:|:-----------------------------------|----------:|--------------------------------:|---------------------------------------:|-----------:|
|  0 | SVM {'C': 10, 'max_iter': 30000}   |         7 |                         92.9362 |                                95.5581 | 158.113    |
|  1 | SVM {'C': 100, 'max_iter': 30000}  |         7 |                         94.9733 |                                96.3479 | 133.883    |
|  2 | SVM {'C': 1000, 'max_iter': 30000} |         7 |                         94.319  |                                94.8947 | 151.229    |
|  3 | KNN {'n_neighbors': 3}             |         7 |                         94.1888 |                                95.285  |   0.332991 |
|  4 | KNN {'n_neighbors': 5}             |         7 |                         94.4989 |                                95.6145 |   0.297394 |
|  5 | KNN {'n_neighbors': 10}            |         7 |                         94.4719 |                                95.4222 |   0.30521  |
|  6 | DT {'max_depth': None}             |         7 |                         92.159  |                                93.5755 |   0.508481 |
|  7 | DT {'max_depth': 10}               |         7 |                         93.8106 |                                95.4545 |   0.44297  |
|  8 | DT {'max_depth': 15}               |         7 |                         93.0085 |                                94.1934 |   0.498818 |
