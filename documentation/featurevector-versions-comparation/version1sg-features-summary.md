# Version 8 Feature Vector Evaluation Summary

The training set is always 0.8 and the testing set is always 0.2 of the dataset.

## Bellevue 116th NE12th

- Dataset size: 3826 
- Dataset size (after OPTICS clustering): 3037
- OPTICS parameters:
    - min-samples: 100 
    - max-eps: 0.25
- K-Means MSE threshold: 0.2
- Number of Clusters: 9 
- Number of reduced clusters: 4
- Training set size: 2678 
- Testing set size: 670 

### Version 1

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.906977 | 0.980322 | 0.995656 |            0.842836 |
| KNN | 0.946077 | 0.993355 | 0.995656 |            0.915429 |
| DT  | 0.93381  | 0.953488 | 0.962688 |            0.924747 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.935344 | 0.994633 | 1        |            0.925492 |
| KNN | 0.966777 | 0.995911 | 0.995911 |            0.962895 |
| DT  | 0.939944 | 0.950933 | 0.950933 |            0.949377 |

### Version 1SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.906636 | 0.980967 | 0.994599 |            0.841478 |
| KNN | 0.945216 | 0.99357  | 0.99537  |            0.914054 |
| DT  | 0.936214 | 0.954218 | 0.964506 |            0.925586 |                  

#### Pooling Cluster (K-Means)
                                                                                               
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.934671 | 0.994342 | 0.999743 |            0.924868 |
| KNN | 0.965535 | 0.995885 | 0.995885 |            0.961524 |
| DT  | 0.941615 | 0.951646 | 0.951646 |            0.951529 |

## Bellevue 150th Newport

- Dataset size: 42192
- Dataset size (after OPTICS clustering): 39326
- OPTICS parameters
    - min-samples: 200
    - max-eps: 0.1
- K-Means MSE threshold: 0.2
- Number of Clusters: 11
- Number of reduced clusters: 4
- Training set size: 31460
- Testing set size: 7866

### Version 1

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.840585 | 0.96453  | 0.99824  |            0.744527 |
| KNN | 0.932723 | 0.985916 | 0.989271 |            0.880916 |
| DT  | 0.892352 | 0.918782 | 0.92078  |            0.869322 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.88864  | 0.979113 | 0.999857 |            0.853813 |
| KNN | 0.943381 | 0.987629 | 0.990151 |            0.930028 |
| DT  | 0.902224 | 0.919971 | 0.920114 |            0.894306 |

### Version 1SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.829732 | 0.9513   | 0.997017 |            0.750907 |
| KNN | 0.912967 | 0.984842 | 0.989211 |            0.867351 |
| DT  | 0.876593 | 0.904349 | 0.90682  |            0.85257  |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.882229 | 0.970648 | 0.999759 |            0.870698 |
| KNN | 0.927161 | 0.98674  | 0.990266 |            0.916047 |
| DT  | 0.88467  | 0.905946 | 0.906066 |            0.882466 |

## Bellevue 150th Eastgate

- Dataset size: 32261 
- Dataset size (after OPTICS clustering): 32052
- OPTICS parameters:
    - min-samples: 200 
    - max-eps: 0.16
- K-Means MSE threshold: 0.2
- Number of Clusters: 11 
- Number of reduced clusters: 4
- Training set size: 25641 
- Testing set size: 6411

### Version 1

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.937239 | 0.992192 | 0.998823 |            0.905155 |
| KNN | 0.96759  | 0.995668 | 0.997299 |            0.957144 |
| DT  | 0.973794 | 0.981709 | 0.982485 |            0.967585 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.981201 | 0.996791 | 0.999866 |            0.978895 |
| KNN | 0.987833 | 0.997968 | 0.998075 |            0.985177 |
| DT  | 0.97727  | 0.9808   | 0.9808   |            0.97698  |

### Version 1SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.937321 | 0.992249 | 0.99882  |            0.905287 |
| KNN | 0.967708 | 0.995628 | 0.997211 |            0.957386 |
| DT  | 0.973582 | 0.981923 | 0.982862 |            0.967249 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.981172 | 0.996808 | 0.999866 |            0.978819 |
| KNN | 0.987904 | 0.997881 | 0.998015 |            0.985259 |
| DT  | 0.977632 | 0.981199 | 0.981199 |            0.976481 |

## Bellevue 150 SE 38th

- Dataset size: 21921 
- Dataset size (after OPTICS clustering): 21469
- OPTICS parameters:
    - min-samples: 100
    - max-eps: 0.15
- K-Means MSE threshold: 0.2
- Number of Clusters: 13
- Number of reduced clusters: 5
- Training set size: 17175
- Testing set size: 4294

### Version 1

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.859212 | 0.970398 | 0.993577 |            0.785164 |
| KNN | 0.944227 | 0.99238  | 0.992619 |            0.926505 |
| DT  | 0.912551 | 0.931421 | 0.93178  |            0.911171 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.873295 | 0.975505 | 0.995771 |            0.827411 |
| KNN | 0.952725 | 0.993338 | 0.993417 |            0.947471 |
| DT  | 0.91654  | 0.933456 | 0.933496 |            0.91391  |

### Version 1SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.858888 | 0.970428 | 0.993692 |            0.786622 |
| KNN | 0.94415  | 0.992406 | 0.992647 |            0.926266 |
| DT  | 0.913854 | 0.932377 | 0.932899 |            0.91272  |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.87271  | 0.97549  | 0.995982 |            0.827255 |
| KNN | 0.952547 | 0.99337  | 0.99341  |            0.947222 |
| DT  | 0.921649 | 0.934667 | 0.934707 |            0.914433 |

## Bellevue NE8th

- Dataset size: 33580
- Dataset size (after OPTICS clustering): 32023
- OPTICS parameters:
    - min-samples: 400 
    - max-eps: 0.15
- K-Means MSE threshold: 0.2
- Number of Clusters: 11 
- Number of reduced clusters: 4
- Training set size: 25620
- Testing set size: 6406

### Version 1

#### Clusters (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.925671 | 0.986668 | 0.998578 |            0.807467 |
| KNN | 0.958798 | 0.990799 | 0.997157 |            0.922475 |
| DT  | 0.95971  | 0.969903 | 0.971701 |            0.92991  |

#### Pooled clusters (K-Means MSE Search)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.973283 | 0.999732 | 0.999839 |            0.972602 |
| KNN | 0.987983 | 0.998203 | 0.998203 |            0.987715 |
| DT  | 0.971513 | 0.974759 | 0.974759 |            0.973044 |

#### Version 1SG

#### Clusters (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.923797 | 0.985467 | 0.998257 |            0.807016 |
| KNN | 0.958413 | 0.990723 | 0.997211 |            0.918676 |
| DT  | 0.95667  | 0.969755 | 0.973214 |            0.924096 |

#### Pooled clusters (K-Means MSE Search)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.970613 | 0.999517 | 0.999812 |            0.969476 |
| KNN | 0.986513 | 0.997345 | 0.997345 |            0.986345 |
| DT  | 0.969621 | 0.971873 | 0.9719   |            0.971005 |