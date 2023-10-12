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
| SVM | 0.938553 | 0.985435 | 0.993931 |            0.852172 |
| KNN | 0.958883 | 0.996662 | 0.997572 |            0.909135 |
| DT  | 0.932484 | 0.949021 | 0.956152 |            0.889312 |

#### Pooling Cluster (K-Means)
                                                                                               
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.959945 | 0.99469  | 1        |            0.956948 |
| KNN | 0.973904 | 0.998179 | 0.998179 |            0.97135  |
| DT  | 0.935518 | 0.943104 | 0.943104 |            0.949885 |

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
| SVM | 0.923393 | 0.990771 | 0.997493 |            0.912527 |
| KNN | 0.929538 | 0.981719 | 0.982606 |            0.926558 |
| DT  | 0.908262 | 0.928695 | 0.930581 |            0.906215 |


#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.932289 | 0.993433 | 0.998003 |            0.931542 |
| KNN | 0.935306 | 0.980454 | 0.980721 |            0.935695 |
| DT  | 0.911435 | 0.919355 | 0.919355 |            0.920525 |

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
| SVM | 0.819671 | 0.964036 | 0.989392 |            0.733208 |
| KNN | 0.825816 | 0.972345 | 0.983407 |            0.778466 |
| DT  | 0.769064 | 0.798669 | 0.805189 |            0.721125 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.833672 | 0.971009 | 0.994843 |            0.847698 |
| KNN | 0.831721 | 0.975979 | 0.984075 |            0.833506 |
| DT  | 0.760541 | 0.801047 | 0.801689 |            0.773584 |

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
| SVM | 0.843511 | 0.983816 | 0.995078 |            0.792782 |
| KNN | 0.889216 | 0.97692  | 0.978023 |            0.868552 |
| DT  | 0.828161 | 0.856487 | 0.856905 |            0.832278 |

#### Pooling Cluster (K-Means)

|     |   Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|--------:|---------:|---------:|--------------------:|
| SVM | 0.85303 | 0.988034 | 0.998285 |            0.859268 |
| KNN | 0.8964  | 0.978792 | 0.979701 |            0.907846 |
| DT  | 0.82983 | 0.861437 | 0.86153  |            0.845651 |

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

#### Pooled clusters (K-Means MSE Search)
