# Y coordinate scaling comparison

## V1 scaling factor 1.5

## V7 weights 1, 1.5, 100, 150, 2, 3, 200, 300

|    | classifier   | version   |   balanced_test_score (percent) |   balanced_pooled_test_score (percent) |   time (s) |
|---:|:-------------|:----------|--------------------------------:|---------------------------------------:|-----------:|
|  0 | KNN          | 1         |                         95.3102 |                                96.0973 |   0.320985 |
|  1 | DT           | 1         |                         95.3027 |                                96.0205 |   0.752156 |
|  2 | SVM          | 1         |                         80.6412 |                                90.6312 | 209.281    |
|  3 | KNN          | 1_SY      |                         95.1891 |                                96.0963 |   0.300297 |
|  4 | DT           | 1_SY      |                         95.4236 |                                95.985  |   0.839526 |
|  5 | SVM          | 1_SY      |                         82.9569 |                                91.4253 | 208.73     |
|  6 | KNN          | 7         |                         94.4633 |                                95.6485 |   0.323509 |
|  7 | DT           | 7         |                         92.1248 |                                93.569  |   0.489379 |
|  8 | SVM          | 7         |                         86.1363 |                                93.0888 | 273.403    |
|  9 | KNN          | 7_SY      |                         94.8767 |                                95.9078 |   0.338065 |
| 10 | DT           | 7_SY      |                         92.1251 |                                93.4502 |   0.49361  |
| 11 | SVM          | 7_SY      |                         88.0361 |                                93.6767 | 278.111    |