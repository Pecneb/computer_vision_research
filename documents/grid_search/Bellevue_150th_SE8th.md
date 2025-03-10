| classifier   | version   | split_0 (percent)   | split_1 (percent)   | split_2 (percent)   | split_3 (percent)   | split_4 (percent)   | mean (percent)   | std (percent)   |
|--------------|-----------|---------------------|---------------------|---------------------|---------------------|---------------------|------------------|-----------------|

|    | classifier                        |   version |   balanced_test_score (percent) |   balanced_pooled_test_score (percent) |   time (s) |
|---:|:----------------------------------|----------:|--------------------------------:|---------------------------------------:|-----------:|
|  0 | SVM {'C': 0.1, 'max_iter': 5000}  |         7 |                         65.3767 |                                77.4549 | 192.001    |
|  1 | SVM {'C': 0.1, 'max_iter': 15000} |         7 |                         64.248  |                                73.6379 | 406.585    |
|  2 | SVM {'C': 0.1, 'max_iter': 30000} |         7 |                         64.248  |                                73.6379 | 365.228    |
|  3 | SVM {'C': 1, 'max_iter': 5000}    |         7 |                         57.754  |                                67.8873 | 183.674    |
|  4 | SVM {'C': 1, 'max_iter': 15000}   |         7 |                         77.4167 |                                91.7891 | 316.58     |
|  5 | SVM {'C': 1, 'max_iter': 30000}   |         7 |                         77.4167 |                                91.7891 | 264.942    |
|  6 | SVM {'C': 10, 'max_iter': 5000}   |         7 |                         72.2862 |                                86.177  | 124.245    |
|  7 | SVM {'C': 10, 'max_iter': 15000}  |         7 |                         81.5903 |                                96.198  | 165.87     |
|  8 | SVM {'C': 10, 'max_iter': 30000}  |         7 |                         81.5903 |                                96.198  | 167.621    |
|  9 | KNN {'n_neighbors': 3}            |         7 |                         86.8867 |                                96.4625 |   0.311477 |
| 10 | KNN {'n_neighbors': 5}            |         7 |                         87.6628 |                                97.199  |   0.382378 |
| 11 | KNN {'n_neighbors': 10}           |         7 |                         87.3386 |                                97.6158 |   0.390004 |
| 12 | DT {'max_depth': None}            |         7 |                         83.4801 |                                93.6874 |   0.53578  |
| 13 | DT {'max_depth': 5}               |         7 |                         84.9596 |                                95.6982 |   0.404754 |
| 14 | DT {'max_depth': 10}              |         7 |                         85.9581 |                                96.2645 |   0.515927 |

| classifier   | version   | split_0 (percent)   | split_1 (percent)   | split_2 (percent)   | split_3 (percent)   | split_4 (percent)   | mean (percent)   | std (percent)   |
|--------------|-----------|---------------------|---------------------|---------------------|---------------------|---------------------|------------------|-----------------|
|    | classifier                         |   version |   balanced_test_score (percent) |   balanced_pooled_test_score (percent) |   time (s) |
|---:|:-----------------------------------|----------:|--------------------------------:|---------------------------------------:|-----------:|
|  0 | SVM {'C': 10, 'max_iter': 30000}   |         7 |                         81.5903 |                                96.198  | 163.361    |
|  1 | SVM {'C': 100, 'max_iter': 30000}  |         7 |                         85.6846 |                                97.2301 |  96.5215   |
|  2 | SVM {'C': 1000, 'max_iter': 30000} |         7 |                         81.4415 |                                89.1033 |  93.3117   |
|  3 | KNN {'n_neighbors': 3}             |         7 |                         86.8867 |                                96.4625 |   0.438876 |
|  4 | KNN {'n_neighbors': 5}             |         7 |                         87.6628 |                                97.199  |   0.399673 |
|  5 | KNN {'n_neighbors': 10}            |         7 |                         87.3386 |                                97.6158 |   0.408655 |
|  6 | DT {'max_depth': None}             |         7 |                         83.9451 |                                94.2173 |   0.571382 |
|  7 | DT {'max_depth': 10}               |         7 |                         86.202  |                                96.8052 |   0.536129 |
|  8 | DT {'max_depth': 15}               |         7 |                         86.0375 |                                96.3854 |   0.545063 |
