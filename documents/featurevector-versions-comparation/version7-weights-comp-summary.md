# Feature Vector version 7 with different weights vectors comparison

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

### Version 7x0.5

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.93567  | 0.985435 | 0.994993 |            0.85893  |
| KNN | 0.951752 | 0.99651  | 0.997724 |            0.895916 |
| DT  | 0.934304 | 0.950235 | 0.956607 |            0.895334 |

#### Pooling Cluster (K-Means)
                                                                                               
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.955849 | 0.991352 | 0.999241 |            0.952535 |
| KNN | 0.966166 | 0.998179 | 0.998331 |            0.963889 |
| DT  | 0.939766 | 0.946594 | 0.946746 |            0.95003  |

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

### Version 7x0.5

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.804208 | 0.947333 | 0.982254 |            0.711396 |
| KNN | 0.81774  | 0.972258 | 0.981576 |            0.777054 |
| DT  | 0.760382 | 0.788092 | 0.796616 |            0.731468 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.817168 | 0.953718 | 0.990947 |            0.818157 |
| KNN | 0.824834 | 0.975297 | 0.983228 |            0.830275 |
| DT  | 0.761579 | 0.790697 | 0.791131 |            0.772611 |

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

### Version 7x0.5

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.915006 | 0.989241 | 0.996828 |            0.896077 |
| KNN | 0.926245 | 0.985044 | 0.98663  |            0.9207   |
| DT  | 0.908489 | 0.930469 | 0.933243 |            0.90788  |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.923354 | 0.991688 | 0.997725 |            0.918695 |
| KNN | 0.932626 | 0.984799 | 0.985661 |            0.927768 |
| DT  | 0.911172 | 0.91826  | 0.918287 |            0.919085 |

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

### Version 7x0.5

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.784977 | 0.971285 | 0.994188 |            0.688185 |
| KNN | 0.879363 | 0.978218 | 0.981981 |            0.855176 |
| DT  | 0.831711 | 0.860621 | 0.860937 |            0.835702 |


#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.793968 | 0.976169 | 0.99772  |            0.753074 |
| KNN | 0.886278 | 0.979784 | 0.983686 |            0.896083 |
| DT  | 0.845837 | 0.865312 | 0.865312 |            0.849092 |

## Bellevue NE8th

- Dataset size: 33580
- Dataset size (after OPTICS clustering): 32023
- OPTICS parameters:
    - min-samples: 400 
    - max-eps: 0.15
- K-Means MSE threshold: 0.2
- Number of Clusters: 11 
- Number of reduced clusters: 4
- Training set size: 16011
- Testing set size: 16012

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

#### Version 7x0.5

#### Clusters (OPTICS)



#### Pooled clusters (K-Means MSE Search)

