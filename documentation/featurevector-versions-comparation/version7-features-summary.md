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

### Version 7

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.938401 | 0.985738 | 0.993628 |            0.852021 |
| KNN | 0.958732 | 0.996662 | 0.998028 |            0.910633 |
| DT  | 0.93476  | 0.949021 | 0.955849 |            0.894897 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.959945 | 0.99469  | 1        |            0.956962 |
| KNN | 0.973297 | 0.998179 | 0.998179 |            0.97051  |
| DT  | 0.937642 | 0.946897 | 0.946897 |            0.949974 |

### Version 7SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |   
|:----|---------:|---------:|---------:|--------------------:|   
| SVM | 0.910939 | 0.97618  | 0.988773 |            0.804453 |   
| KNN | 0.922622 | 0.989379 | 0.993324 |            0.839765 |   
| DT  | 0.913215 | 0.934911 | 0.944318 |            0.859154 |   

#### Pooling Cluster (K-Means)
                                                                                               
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy | 
|:----|---------:|---------:|---------:|--------------------:| 
| SVM | 0.942346 | 0.991504 | 0.998938 |            0.938745 | 
| KNN | 0.952814 | 0.995752 | 0.996207 |            0.949895 | 
| DT  | 0.927173 | 0.93476  | 0.93476  |            0.942683 | 

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

### Version 7

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.820446 | 0.963849 | 0.988217 |            0.734215 |
| KNN | 0.826431 | 0.972719 | 0.9833   |            0.790677 |
| DT  | 0.77016  | 0.801689 | 0.811227 |            0.717202 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.83434  | 0.96978  | 0.99471  |            0.848615 |
| KNN | 0.833031 | 0.975739 | 0.984182 |            0.833322 |
| DT  | 0.775771 | 0.805162 | 0.805857 |            0.779292 |

### Version 7SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.796051 | 0.95765  | 0.977983 |            0.704049 |
| KNN | 0.793245 | 0.96457  | 0.983487 |            0.733174 |
| DT  | 0.748277 | 0.786245 | 0.792845 |            0.698147 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.826324 | 0.970048 | 0.990247 |            0.836983 |
| KNN | 0.818923 | 0.972666 | 0.98656  |            0.81419  |
| DT  | 0.760968 | 0.790947 | 0.791348 |            0.755141 |

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

### Version 7

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.923348 | 0.990948 | 0.997493 |            0.912401 |
| KNN | 0.928518 | 0.981053 | 0.982185 |            0.924715 |
| DT  | 0.910103 | 0.92834  | 0.930403 |            0.908684 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.931979 | 0.993455 | 0.998092 |            0.931072 |
| KNN | 0.934685 | 0.9795   | 0.979833 |            0.934872 |
| DT  | 0.911324 | 0.918401 | 0.918445 |            0.921329 |

### Version 7SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.899765 | 0.986822 | 0.994076 |            0.881616 |
| KNN | 0.904224 | 0.982296 | 0.986844 |            0.895709 |
| DT  | 0.895771 | 0.921152 | 0.924324 |            0.892185 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.919976 | 0.99259  | 0.99716  |            0.916482 |
| KNN | 0.926077 | 0.984292 | 0.986134 |            0.925105 |
| DT  | 0.90276  | 0.914141 | 0.914186 |            0.914181 |

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

### Version 7

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.858462 | 0.985099 | 0.993966 |            0.814137 |
| KNN | 0.893087 | 0.9787   | 0.979545 |            0.871655 |
| DT  | 0.836636 | 0.85853  | 0.858942 |            0.823753 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.868792 | 0.990835 | 0.998469 |            0.884162 |
| KNN | 0.90152  | 0.982791 | 0.983339 |            0.909511 |
| DT  | 0.851628 | 0.865341 | 0.865341 |            0.855211 |

### Version 7SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.787424 | 0.970914 | 0.992223 |            0.652133 |
| KNN | 0.866618 | 0.975057 | 0.981193 |            0.809289 |
| DT  | 0.81233  | 0.846801 | 0.847951 |            0.789499 |


#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.80218  | 0.979654 | 0.997961 |            0.760918 |
| KNN | 0.879873 | 0.978533 | 0.983594 |            0.886381 |
| DT  | 0.833519 | 0.853883 | 0.853938 |            0.832248 |

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

### Version 7

#### Clusters (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.955113 | 0.99194  | 0.99745  |            0.844304 |
| KNN | 0.961105 | 0.988761 | 0.994432 |            0.895838 |
| DT  | 0.933872 | 0.951473 | 0.960268 |            0.877884 |

#### Pooled clusters (K-Means MSE Search)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.967715 | 0.99815  | 0.999426 |            0.961862 |
| KNN | 0.97415  | 0.991872 | 0.991886 |            0.970819 |
| DT  | 0.947545 | 0.951833 | 0.951833 |            0.947075 |

#### Version 7SG

#### Clusters (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.935503 | 0.985179 | 0.995172 |            0.805769 |
| KNN | 0.938126 | 0.98368  | 0.993614 |            0.844418 |
| DT  | 0.915527 | 0.938725 | 0.948221 |            0.850765 |

#### Pooled clusters (K-Means MSE Search)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.958735 | 0.996715 | 0.998705 |            0.951708 |
| KNN | 0.965676 | 0.99248  | 0.992748 |            0.961017 |
| DT  | 0.93746  | 0.943033 | 0.943047 |            0.93793  |