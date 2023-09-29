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

### Version 8

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |                                 
|:----|---------:|---------:|---------:|--------------------:|                                 
| SVM | 0.904737 | 0.988579 | 0.997274 |            0.841324 |                                 
| KNN | 0.938871 | 0.993381 | 0.995977 |            0.903359 |                                 
| DT  | 0.879689 | 0.912005 | 0.922777 |            0.86119  |                                 

#### Pooling Cluster (K-Means)
                                                                                               
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |                                 
|:----|---------:|---------:|---------:|--------------------:|         
| SVM | 0.942245 | 0.998053 | 1        |            0.933471 |                                 
| KNN | 0.964828 | 0.995717 | 0.995717 |            0.960794 |                            
| DT  | 0.909799 | 0.920571 | 0.920571 |            0.918294 |  

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
| SVM | 0.759019 | 0.951304 | 0.995682 |            0.665773 |
| KNN | 0.929598 | 0.987283 | 0.99094  |            0.87873  |
| DT  | 0.888852 | 0.923723 | 0.929904 |            0.866283 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.832629 | 0.971971 | 0.999693 |            0.731695 |
| KNN | 0.941418 | 0.987519 | 0.989926 |            0.927759 |
| DT  | 0.901427 | 0.919571 | 0.919736 |            0.900655 |

### Version 8

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.740919 | 0.942523 | 0.998252 |            0.677885 |
| KNN | 0.880289 | 0.978436 | 0.987147 |            0.843011 |
| DT  | 0.812982 | 0.85496  | 0.867593 |            0.772443 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.819771 | 0.961661 | 0.999795 |            0.758513 |
| KNN | 0.906814 | 0.981807 | 0.987934 |            0.897564 |
| DT  | 0.838358 | 0.861781 | 0.862096 |            0.846406 |

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
| SVM | 0.941117 | 0.993047 | 0.998823 |            0.915172 |
| KNN | 0.967483 | 0.995401 | 0.997059 |            0.957237 |
| DT  | 0.973634 | 0.981709 | 0.982405 |            0.967479 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.982592 | 0.997674 | 0.999893 |            0.979288 |
| KNN | 0.987833 | 0.997834 | 0.997941 |            0.985165 |
| DT  | 0.976789 | 0.980666 | 0.980666 |            0.976137 |

### Version 8

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.935545 | 0.994586 | 0.998963 |            0.908089 |
| KNN | 0.952352 | 0.993064 | 0.995488 |            0.940579 |
| DT  | 0.92671  | 0.944622 | 0.946292 |            0.9222   |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.985226 | 0.999057 | 0.999811 |            0.983225 |
| KNN | 0.985253 | 0.996458 | 0.996606 |            0.982448 |
| DT  | 0.955719 | 0.959598 | 0.959611 |            0.962662 |

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

### Version 8

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.873066 | 0.982214 | 0.994454 |            0.809515 |
| KNN | 0.928142 | 0.990258 | 0.991793 |            0.900995 |
| DT  | 0.878899 | 0.901576 | 0.904318 |            0.856719 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.895682 | 0.989091 | 0.99781  |            0.879704 |
| KNN | 0.943348 | 0.992714 | 0.992796 |            0.939998 |
| DT  | 0.896132 | 0.910806 | 0.910806 |            0.885561 |

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

#### Version 8

#### Clusters (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.902323 | 0.973617 | 0.990782 |            0.787438 |
| KNN | 0.949089 | 0.988411 | 0.996252 |            0.909999 |
| DT  | 0.921408 | 0.937659 | 0.941963 |            0.874402 |

#### Pooled clusters (K-Means MSE Search)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.956665 | 0.98955  | 0.999775 |            0.953411 |
| KNN | 0.986663 | 0.997801 | 0.997815 |            0.986421 |
| DT  | 0.95481  | 0.958664 | 0.958664 |            0.956196 |