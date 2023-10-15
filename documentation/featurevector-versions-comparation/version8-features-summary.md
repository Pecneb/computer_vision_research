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
| SVM | 0.910636 | 0.981793 | 0.995448 |            0.798054 |
| KNN | 0.948415 | 0.9956   | 0.998028 |            0.892796 |
| DT  | 0.913367 | 0.933697 | 0.942497 |            0.844374 |            

#### Pooling Cluster (K-Means)
                                                                             
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.940525 | 1        | 1        |            0.937148 |
| KNN | 0.971325 | 0.999241 | 0.999241 |            0.968384 |
| DT  | 0.935063 | 0.941284 | 0.941284 |            0.943183 |

### Version 8SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.911243 | 0.982097 | 0.995448 |            0.799138 |
| KNN | 0.948111 | 0.9956   | 0.997876 |            0.89246  |
| DT  | 0.920801 | 0.939463 | 0.949628 |            0.864719 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.940828 | 1        | 1        |            0.937494 |
| KNN | 0.971476 | 0.999241 | 0.999241 |            0.968515 |
| DT  | 0.942497 | 0.950083 | 0.950083 |            0.9475   |
                
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
| SVM | 0.793379 | 0.953001 | 0.989766 |            0.695642 |
| KNN | 0.813392 | 0.956528 | 0.967135 |            0.748111 |
| DT  | 0.756266 | 0.787901 | 0.798295 |            0.700966 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.831347 | 0.965959 | 0.997141 |            0.843364 |
| KNN | 0.824587 | 0.960215 | 0.968044 |            0.821369 |
| DT  | 0.765484 | 0.793406 | 0.793566 |            0.765015 |

### Version 8SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.793352 | 0.952814 | 0.989847 |            0.69616  |
| KNN | 0.812029 | 0.9561   | 0.966948 |            0.746632 |
| DT  | 0.757121 | 0.786485 | 0.794394 |            0.703863 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.831267 | 0.965853 | 0.997141 |            0.843699 |
| KNN | 0.822182 | 0.959787 | 0.967803 |            0.819683 |
| DT  | 0.762171 | 0.792551 | 0.792684 |            0.763897 |

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
| SVM | 0.920708 | 0.994121 | 0.997338 |            0.914078 |
| KNN | 0.919599 | 0.966965 | 0.967742 |            0.916089 |
| DT  | 0.902937 | 0.923348 | 0.925833 |            0.900518 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.937592 | 0.998025 | 0.999068 |            0.936767 |
| KNN | 0.928384 | 0.963771 | 0.963771 |            0.92985  |
| DT  | 0.909105 | 0.916515 | 0.916515 |            0.918583 |

### Version 8SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.920708 | 0.99421  | 0.997316 |            0.913963 |
| KNN | 0.919865 | 0.967675 | 0.968385 |            0.916527 |
| DT  | 0.901518 | 0.924968 | 0.927475 |            0.901212 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.937636 | 0.998048 | 0.999068 |            0.936816 |
| KNN | 0.928473 | 0.96437  | 0.96437  |            0.929903 |
| DT  | 0.906687 | 0.917491 | 0.917491 |            0.919303 |

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
| SVM | 0.855106 | 0.983751 | 0.995319 |            0.750374 |
| KNN | 0.880828 | 0.96132  | 0.961839 |            0.846088 |
| DT  | 0.833046 | 0.854634 | 0.855514 |            0.813651 | 

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.869093 | 0.98986  | 0.998211 |            0.848961 |
| KNN | 0.89158  | 0.963424 | 0.963591 |            0.901828 |
| DT  | 0.846171 | 0.862429 | 0.862447 |            0.853031 |

### Version 8SG

#### Cluster (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.878391 | 0.982356 | 0.994835 |            0.809594 |
| KNN | 0.888561 | 0.964667 | 0.965261 |            0.853947 |
| DT  | 0.831539 | 0.859696 | 0.861113 |            0.811872 |

#### Pooling Cluster (K-Means)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.892607 | 0.990584 | 0.99936  |            0.893759 |
| KNN | 0.900674 | 0.968895 | 0.969032 |            0.908002 |
| DT  | 0.841115 | 0.869158 | 0.869181 |            0.854284 |

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
| SVM | 0.942298 | 0.987029 | 0.994422 |            0.812574 |
| KNN | 0.952797 | 0.986167 | 0.993064 |            0.884943 |
| DT  | 0.920623 | 0.940818 | 0.949681 |            0.869979 |

#### Pooled clusters (K-Means MSE Search)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.963295 | 0.997756 | 0.999528 |            0.956216 |
| KNN | 0.970514 | 0.990095 | 0.990114 |            0.967491 |
| DT  | 0.939591 | 0.945491 | 0.945496 |            0.941532 |

#### Version 8SG

#### Clusters (OPTICS)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.937033 | 0.986912 | 0.994175 |            0.80917  |
| KNN | 0.94705  | 0.981915 | 0.991409 |            0.8699   |
| DT  | 0.922215 | 0.940701 | 0.952948 |            0.854855 |

#### Pooled clusters (K-Means MSE Search)

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.958749 | 0.997563 | 0.999793 |            0.950038 |
| KNN | 0.965269 | 0.984621 | 0.984621 |            0.960765 |
| DT  | 0.941201 | 0.944394 | 0.944394 |            0.9403   |