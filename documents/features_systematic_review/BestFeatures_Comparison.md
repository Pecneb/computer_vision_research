# Comparison of the best performing feature vectors, with longer strides and savgol windows

## Bellevue 150th SE38

## Max Stride 40, SavGol window length 15

|    | classifier                                                                                                                                                                      | version   |   split_0 (percent) |   split_1 (percent) |   split_2 (percent) |   split_3 (percent) |   split_4 (percent) |   mean (percent) |   std (percent) |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|--------------------:|--------------------:|--------------------:|--------------------:|--------------------:|-----------------:|----------------:|
|  0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs    |             88.3364 |             85.9435 |             88.7115 |             87.9971 |             84.6996 |          87.1376 |         1.55066 |
|  1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs    |             90.1291 |             87.7482 |             88.7031 |             89.2189 |             85.9101 |          88.3419 |         1.43944 |
|  2 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRm    |             87.031  |             84.3783 |             86.6422 |             87.2659 |             83.1426 |          85.692  |         1.63694 |
|  3 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRm    |             88.9729 |             87.2363 |             88.3788 |             88.1265 |             84.8708 |          87.5171 |         1.43629 |
|  4 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRsRm  |             88.6887 |             86.0054 |             88.3513 |             88.4147 |             84.3028 |          87.1526 |         1.72205 |
|  5 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRsRm  |             90.577  |             87.8438 |             86.058  |             89.3117 |             86.3824 |          88.0346 |         1.71858 |

|    | classifier                                                                                                                                                                      | version   |   balanced_test_score (percent) |   balanced_pooled_test_score (percent) |   time (s) |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|--------------------------------:|---------------------------------------:|-----------:|
|  0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs    |                         86.4438 |                                97.2571 |   0.519946 |
|  1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs    |                         85.0709 |                                98.0411 | 389.531    |
|  2 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRm    |                         85.5065 |                                97.0368 |   0.503638 |
|  3 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRm    |                         84.4344 |                                98.1032 | 520.185    |
|  4 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRsRm  |                         86.7017 |                                97.1165 |   0.569127 |
|  5 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRsRm  |                         86.0373 |                                98.1608 | 400.023    |
Total time: 5230.040454139 seconds

## Bellvue 150th Newport

## Max Stride 40, SavGol window length 15

|    | classifier                                                                                                                                                                      | version   |   split_0 (percent) |   split_1 (percent) |   split_2 (percent) |   split_3 (percent) |   split_4 (percent) |   mean (percent) |   std (percent) |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|--------------------:|--------------------:|--------------------:|--------------------:|--------------------:|-----------------:|----------------:|
|  0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs    |             94.1224 |             93.7721 |             93.3145 |             92.9897 |             94.3874 |          93.7172 |        0.511432 |
|  1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs    |             95.3814 |             94.1383 |             94.8738 |             93.0584 |             94.1021 |          94.3108 |        0.788025 |
|  2 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRm    |             93.1815 |             92.7279 |             92.8116 |             92.6184 |             93.4085 |          92.9496 |        0.297423 |
|  3 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRm    |             94.8425 |             93.3283 |             93.8438 |             92.7901 |             93.348  |          93.6306 |        0.69164  |
|  4 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRsRm  |             94.3436 |             93.8451 |             93.6593 |             93.478  |             94.5522 |          93.9756 |        0.408052 |
|  5 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRsRm  |             95.4408 |             94.581  |             95.4776 |             93.1396 |             94.5948 |          94.6468 |        0.848446 |

|    | classifier                                                                                                                                                                      | version   |   balanced_test_score (percent) |   balanced_pooled_test_score (percent) |   time (s) |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|--------------------------------:|---------------------------------------:|-----------:|
|  0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRs    |                         94.0946 |                                95.0495 |   0.373804 |
|  1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRs    |                         95.7602 |                                96.8357 | 585.238    |
|  2 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRm    |                         93.2412 |                                95.3034 |   0.351828 |
|  3 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRm    |                         94.5184 |                                96.4547 | 397.151    |
|  4 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeRsRm  |                         94.3767 |                                95.3263 |   0.424864 |
|  5 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeRsRm  |                         95.9159 |                                96.941  | 493.419    |
Total time: 5413.567890128001 seconds

|    | classifier                                                                                                                                                                      | version   |   split_0 (percent) |   split_1 (percent) |   split_2 (percent) |   split_3 (percent) |   split_4 (percent) |   mean (percent) |   std (percent) |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|--------------------:|--------------------:|--------------------:|--------------------:|--------------------:|-----------------:|----------------:|
|  0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeAe    |             88.9391 |             88.8718 |             88.9824 |             87.9241 |             89.1498 |          88.7734 |        0.434479 |
|  1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeAe    |             91.8988 |             91.2348 |             92.324  |             90.3235 |             91.9822 |          91.5527 |        0.708675 |
|    | classifier                                                                                                                                                                      | version   |   balanced_test_score (percent) |   balanced_pooled_test_score (percent) |    time (s) |
|---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------|--------------------------------:|---------------------------------------:|------------:|
|  0 | KNN {'n_neighbors': 7}                                                                                                                                                          | ReVeAe    |                         89.896  |                                94.1465 |    0.479236 |
|  1 | MLP {'alpha': 0.01, 'hidden_layer_sizes': (100, 100, 100, 100), 'learning_rate': 'adaptive', 'learning_rate_init': 0.001, 'max_iter': 3000, 'solver': 'adam', 'verbose': False} | ReVeAe    |                         93.1245 |                                96.2638 | 1137.64     |