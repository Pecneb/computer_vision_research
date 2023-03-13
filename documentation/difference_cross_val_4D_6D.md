# 0001_2

## 4D cluster features

### classification features v7 stride 15

python3 -m sklearnex classification.py --n_jobs 12 cross-validation -db research_data/0001_2_308min/0001_2_308min_v3_vel10_acc2_clustered_4D.joblib --output research_data/0001_2_308min/tables/0001_2_308min_v7_cluster_4D.xlsx --classification_features_version v7 --stride 15                                          

Training dataset size: 6216                                                                               
Validation dataset size: 2235
                                                              
*Time: 99 s*                                       
                                                               
#### Classifier parameters                                     
                                                               
{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}                                                                                                                      
                                                               
#### Cross-val Basic accuracy                      
                                                               
|    | Split              |       KNN |       SVM |                            
|---:|:-------------------|----------:|----------:|
|  0 | 1                  | 0.744373  | 0.71865   |                                       
|  1 | 2                  | 0.716814  | 0.70716   |
|  2 | 3                  | 0.736122  | 0.753821  |            
|  3 | 4                  | 0.744167  | 0.721641  | 
|  4 | 5                  | 0.738536  | 0.749799  | 
|  5 | Max split          | 0.744373  | 0.753821  |            
|  6 | Mean               | 0.736003  | 0.730214  | 
|  7 | Standart deviation | 0.0101129 | 0.0183281 |                                                                            
                                                               
#### Cross-val Balanced accuracy                    
                                                               
|    | Split              |       KNN |      SVM |                             
|---:|:-------------------|----------:|---------:|  
|  0 | 1                  | 0.762914  | 0.73236  |                                        
|  1 | 2                  | 0.731858  | 0.702185 | 
|  2 | 3                  | 0.752498  | 0.757626 |             
|  3 | 4                  | 0.720529  | 0.678828 |  
|  4 | 5                  | 0.721203  | 0.761219 |  
|  5 | Max split          | 0.762914  | 0.761219 |             
|  6 | Mean               | 0.7378    | 0.726444 |  
|  7 | Standart deviation | 0.0170633 | 0.031851 |             
                                                               
#### Cross-val Top 1 accuracy                       
                                                               
|    | Split              |       KNN |       SVM |                            
|---:|:-------------------|----------:|----------:| 
|  0 | 1                  | 0.744373  | 0.720257  |                                       
|  1 | 2                  | 0.716814  | 0.704747  | 
|  2 | 3                  | 0.736122  | 0.754626  |            
|  3 | 4                  | 0.744167  | 0.721641  | 
|  4 | 5                  | 0.738536  | 0.749799  | 
|  5 | Max split          | 0.744373  | 0.754626  |            
|  6 | Mean               | 0.736003  | 0.730214  |            
|  7 | Standart deviation | 0.0101129 | 0.0189775 |            
                                                               
#### Cross-val Top 2 accuracy                       
                                                                               
|    | Split              |       KNN |        SVM |                           
|---:|:-------------------|----------:|-----------:|                                      
|  0 | 1                  | 0.986334  | 0.987138   |                           
|  1 | 2                  | 0.984714  | 0.966211   |                                      
|  2 | 3                  | 0.985519  | 0.974256   |                           
|  3 | 4                  | 0.96782   | 0.967015   |                           
|  4 | 5                  | 0.991955  | 0.984714   |                           
|  5 | Max split          | 0.991955  | 0.987138   |                           
|  6 | Mean               | 0.983268  | 0.975867   |                           
|  7 | Standart deviation | 0.0081323 | 0.00871217 |                                                      
                                                                                                                                                               
#### Cross-val Top 3 accuracy                                                             
                                                                                          
|    | Split              |        KNN |       SVM |                                                      
|---:|:-------------------|-----------:|----------:|                                      
|  0 | 1                  | 0.986334   | 0.999196  |                                                      
|  1 | 2                  | 0.988737   | 0.986323  |                                      
|  2 | 3                  | 0.989541   | 0.987128  |                                      
|  3 | 4                  | 0.975865   | 0.978278  |                                      
|  4 | 5                  | 0.995977   | 0.992759  |                                                      
|  5 | Max split          | 0.995977   | 0.999196  |                                      
|  6 | Mean               | 0.987291   | 0.988737  |                                                      
|  7 | Standart deviation | 0.00654343 | 0.0069767 |                                      
                                                                                          
#### Test set basic                                                                       
                                                                                          
|    |      KNN |      SVM |                                                              
|---:|---------:|---------:|                                                                              
|  0 | 0.747651 | 0.710515 |                                                                                                                                                         

#### Test set balanced                                                                                    

|    |     KNN |      SVM |                                                                               
|---:|--------:|---------:|                                                                               
|  0 | 0.77575 | 0.745875 |                                                                               

#### Test set top k                                                                                       

|    | Top   |      KNN |      SVM |                                                                      
|---:|:------|---------:|---------:|                                                                      
|  0 | Top_1 | 0.747651 | 0.710515 |                                                                      
|  1 | Top_2 | 0.990604 | 0.975391 |                                                                      
|  2 | Top_3 | 0.992841 | 0.990604 |                                                                      

### classification features v1

python3 -m sklearnex classification.py --n_jobs 12 cross-validation -db research_data/0001_2_308min/0001_2_308min_v3_vel10_acc2_clustered_4D.joblib --output research_data/0001_2_308min/tables/0001_2_308min_v7_cluster_4D.xlsx --classification_features_version v1


Training dataset size: 1757
Validation dataset size: 596

*Time: 100 s*

#### Classifier parameters

{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}

#### Cross-val Basic accuracy

|    | Split              |        KNN |       SVM |
|---:|:-------------------|-----------:|----------:|
|  0 | 1                  | 0.846591   | 0.832386  |
|  1 | 2                  | 0.846591   | 0.857955  |
|  2 | 3                  | 0.849003   | 0.834758  |
|  3 | 4                  | 0.849003   | 0.837607  |
|  4 | 5                  | 0.85755    | 0.85755   |
|  5 | Max split          | 0.85755    | 0.857955  |
|  6 | Mean               | 0.849747   | 0.844051  |
|  7 | Standart deviation | 0.00404757 | 0.0113091 |

#### Cross-val Balanced accuracy

|    | Split              |        KNN |       SVM |
|---:|:-------------------|-----------:|----------:|
|  0 | 1                  | 0.783965   | 0.751117  |
|  1 | 2                  | 0.780168   | 0.782548  |
|  2 | 3                  | 0.783721   | 0.762234  |
|  3 | 4                  | 0.791337   | 0.768758  |
|  4 | 5                  | 0.79708    | 0.783781  |
|  5 | Max split          | 0.79708    | 0.783781  |
|  6 | Mean               | 0.787254   | 0.769688  |
|  7 | Standart deviation | 0.00611142 | 0.0123718 |

#### Cross-val Top 1 accuracy

|    | Split              |        KNN |       SVM |
|---:|:-------------------|-----------:|----------:|
|  0 | 1                  | 0.846591   | 0.832386  |
|  1 | 2                  | 0.846591   | 0.855114  |
|  2 | 3                  | 0.849003   | 0.834758  |
|  3 | 4                  | 0.849003   | 0.831909  |
|  4 | 5                  | 0.85755    | 0.860399  |
|  5 | Max split          | 0.85755    | 0.860399  |
|  6 | Mean               | 0.849747   | 0.842913  |
|  7 | Standart deviation | 0.00404757 | 0.0122721 |

#### Cross-val Top 2 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.985795   | 0.985795   |
|  1 | 2                  | 1          | 0.980114   |
|  2 | 3                  | 0.980057   | 0.982906   |
|  3 | 4                  | 0.977208   | 0.97151    |
|  4 | 5                  | 1          | 0.988604   |
|  5 | Max split          | 1          | 0.988604   |
|  6 | Mean               | 0.988612   | 0.981786   |
|  7 | Standart deviation | 0.00970099 | 0.00586871 |

#### Cross-val Top 3 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.988636   | 0.994318   |
|  1 | 2                  | 1          | 0.988636   |
|  2 | 3                  | 0.985755   | 0.991453   |
|  3 | 4                  | 0.982906   | 0.982906   |
|  4 | 5                  | 1          | 0.994302   |
|  5 | Max split          | 1          | 0.994318   |
|  6 | Mean               | 0.991459   | 0.990323   |
|  7 | Standart deviation | 0.00720492 | 0.00426445 |

#### Test set basic

|    |     KNN |      SVM |
|---:|--------:|---------:|
|  0 | 0.84396 | 0.800336 |

#### Test set balanced

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.820867 | 0.777579 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.84396  | 0.800336 |
|  1 | Top_2 | 0.996644 | 0.989933 |
|  2 | Top_3 | 0.998322 | 0.996644 |


## 6D cluster features

### classification features v7 stride 15

python3 -m sklearnex classification.py --n_jobs 12 cross-validation -db research_data/0001_2_308min/0001_2_308min_v3_vel10_acc2_clustered_min-samples-10_max-eps-01_xi-015.joblib --output research_data/0001_2_308min/tables/0001_2_308min_v7_cluster_6D.xlsx --stride 15                                                 

Training dataset size: 10805
Validation dataset size: 3603

*Time: 352 s*

#### Classifier parameters

{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}

#### Cross-val Basic accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.942619   | 0.930125   |
|  1 | 2                  | 0.941231   | 0.936141   |
|  2 | 3                  | 0.941694   | 0.936603   |
|  3 | 4                  | 0.940305   | 0.938454   |
|  4 | 5                  | 0.938917   | 0.93105    |
|  5 | Max split          | 0.942619   | 0.938454   |
|  6 | Mean               | 0.940953   | 0.934475   |
|  7 | Standart deviation | 0.00126221 | 0.00327997 |

#### Cross-val Balanced accuracy

|    | Split              |       KNN |       SVM |
|---:|:-------------------|----------:|----------:|
|  0 | 1                  | 0.554625  | 0.453707  |
|  1 | 2                  | 0.529921  | 0.498542  |
|  2 | 3                  | 0.538414  | 0.490851  |
|  3 | 4                  | 0.540969  | 0.516948  |
|  4 | 5                  | 0.518967  | 0.477098  |
|  5 | Max split          | 0.554625  | 0.516948  |
|  6 | Mean               | 0.536579  | 0.487429  |
|  7 | Standart deviation | 0.0118559 | 0.0212157 |

#### Cross-val Top 1 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.942619   | 0.930588   |
|  1 | 2                  | 0.941231   | 0.936141   |
|  2 | 3                  | 0.941694   | 0.935215   |
|  3 | 4                  | 0.940305   | 0.938917   |
|  4 | 5                  | 0.938917   | 0.93105    |
|  5 | Max split          | 0.942619   | 0.938917   |
|  6 | Mean               | 0.940953   | 0.934382   |
|  7 | Standart deviation | 0.00126221 | 0.00315756 |

#### Cross-val Top 2 accuracy

|    | Split              |       KNN |        SVM |
|---:|:-------------------|----------:|-----------:|
|  0 | 1                  | 0.976863  | 0.961592   |
|  1 | 2                  | 0.985655  | 0.964368   |
|  2 | 3                  | 0.982416  | 0.965294   |
|  3 | 4                  | 0.98149   | 0.96807    |
|  4 | 5                  | 0.978251  | 0.965294   |
|  5 | Max split          | 0.985655  | 0.96807    |
|  6 | Mean               | 0.980935  | 0.964924   |
|  7 | Standart deviation | 0.0031166 | 0.00207774 |

#### Cross-val Top 3 accuracy

|    | Split              |        KNN |       SVM |
|---:|:-------------------|-----------:|----------:|
|  0 | 1                  | 0.992133   | 0.980565  |
|  1 | 2                  | 0.992596   | 0.984729  |
|  2 | 3                  | 0.993059   | 0.985192  |
|  3 | 4                  | 0.990282   | 0.983341  |
|  4 | 5                  | 0.986118   | 0.982416  |
|  5 | Max split          | 0.993059   | 0.985192  |
|  6 | Mean               | 0.990838   | 0.983248  |
|  7 | Standart deviation | 0.00254133 | 0.0016659 |

#### Test set basic

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.943658 | 0.931168 |

#### Test set balanced

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.619961 | 0.532536 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.943658 | 0.931168 |
|  1 | Top_2 | 0.980294 | 0.965307 |
|  2 | Top_3 | 0.990286 | 0.979184 |