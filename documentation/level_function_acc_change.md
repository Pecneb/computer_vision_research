# 0001_1

## features v4

### Without level function

`$ python3 classification.py --n_jobs 16 cross-validation -db research_data/0001_1_37min/0001_1_37min_v5_clustered.joblib --output research_data/0001_1_37min/tables/0001_1_37min_v5_clustered_2023-02.26.xlsx --classification_features_version v4 --stride 30 --param_set 4`

Training dataset size: 33589
Validation dataset size: 11437

*Time: 106 s*                                                                                                                                                                                                                                                                                                                                                              
                                                                                          
#### Classifier parameters                                                                                                                                                           
                                                                                          
{'KNN': {'n_neighbors': 7}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 8000}}     
                                                                                                                                                                                     
#### Cross-val Basic accuracy                                                                                                                                                        
                                                                                          
|    | Split              |       KNN |        SVM |                            
|---:|:-------------------|----------:|-----------:|                                                                                                                                                                                                                                                                                                                       
|  0 | 1                  | 0.905776  | 0.954153   |
|  1 | 2                  | 0.941054  | 0.958767   |                                                                                                                                 
|  2 | 3                  | 0.922745  | 0.956832   |
|  3 | 4                  | 0.927657  | 0.959661   |                                                                                                                                 
|  4 | 5                  | 0.941343  | 0.95891    |                                                                                                                                 
|  5 | Max split          | 0.941343  | 0.959661   |                                                                                                                                 
|  6 | Mean               | 0.927715  | 0.957665   |
|  7 | Standart deviation | 0.0131886 | 0.00198876 |                            
                                                                                                                                                                                                                                                                                                                                                                           
#### Cross-val Balanced accuracy                                                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                           
|    | Split              |       KNN |        SVM |                                                                                                                                                                                                                                                                                                                       
|---:|:-------------------|----------:|-----------:|                                                                                                                                                                                                                                                                                                                       
|  0 | 1                  | 0.772016  | 0.748226   |                                      
|  1 | 2                  | 0.746497  | 0.749754   |
|  2 | 3                  | 0.741279  | 0.748707   |
|  3 | 4                  | 0.760328  | 0.750919   |                                      
|  4 | 5                  | 0.767514  | 0.7515     |
|  5 | Max split          | 0.772016  | 0.7515     |
|  6 | Mean               | 0.757527  | 0.749821   |
|  7 | Standart deviation | 0.0118589 | 0.00125021 |

#### Cross-val Top 1 accuracy

|    | Split              |       KNN |        SVM |
|---:|:-------------------|----------:|-----------:|
|  0 | 1                  | 0.905776  | 0.954153   |
|  1 | 2                  | 0.941054  | 0.958767   |
|  2 | 3                  | 0.922745  | 0.957874   |
|  3 | 4                  | 0.927657  | 0.959661   |
|  4 | 5                  | 0.941343  | 0.959208   |
|  5 | Max split          | 0.941343  | 0.959661   |
|  6 | Mean               | 0.927715  | 0.957933   |
|  7 | Standart deviation | 0.0131886 | 0.00197981 |

#### Cross-val Top 2 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.939863   | 0.998511   |
|  1 | 2                  | 0.961447   | 0.997916   |
|  2 | 3                  | 0.947306   | 0.999256   |
|  3 | 4                  | 0.960554   | 1          |
|  4 | 5                  | 0.961739   | 1          |
|  5 | Max split          | 0.961739   | 1          |
|  6 | Mean               | 0.954182   | 0.999137   |
|  7 | Standart deviation | 0.00897551 | 0.00082288 |

#### Cross-val Top 3 accuracy

|    | Split              |         KNN |         SVM |
|---:|:-------------------|------------:|------------:|
|  0 | 1                  | 0.999851    | 1           |
|  1 | 2                  | 1           | 1           |
|  2 | 3                  | 1           | 0.999702    |
|  3 | 4                  | 1           | 1           |
|  4 | 5                  | 1           | 1           |
|  5 | Max split          | 1           | 1           |
|  6 | Mean               | 0.99997     | 0.99994     |
|  7 | Standart deviation | 5.95415e-05 | 0.000119083 |

#### Test set basic

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.950861 | 0.979453 |

#### Test set balanced

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.777641 | 0.749757 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.950861 | 0.979453 |
|  1 | Top_2 | 0.980764 | 0.998164 |
|  2 | Top_3 | 1        | 1        |

### With level function

`$ python3 classification.py --n_jobs 16 cross-validation -db research_data/0001_1_37min/0001_1_37min_v5_clustered.joblib --output research_data/0001_1_37min/tables/0001_1_37min_v5_clustered_2023-02.26.xlsx --classification_features_version v4 --stride 30 --param_set 4 --level`

Training dataset size: 5436
Validation dataset size: 904

*Time: 33 s*

#### Classifier parameters

{'KNN': {'n_neighbors': 7}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 8000}}

#### Cross-val Basic accuracy

|    | Split              |      KNN |       SVM |
|---:|:-------------------|---------:|----------:|
|  0 | 1                  | 0.801471 | 0.868566  |
|  1 | 2                  | 0.917203 | 0.853726  |
|  2 | 3                  | 0.685373 | 0.840846  |
|  3 | 4                  | 0.828887 | 0.831647  |
|  4 | 5                  | 0.842686 | 0.885005  |
|  5 | Max split          | 0.917203 | 0.885005  |
|  6 | Mean               | 0.815124 | 0.855958  |
|  7 | Standart deviation | 0.075354 | 0.0191157 |

#### Cross-val Balanced accuracy

|    | Split              |       KNN |      SVM |
|---:|:-------------------|----------:|---------:|
|  0 | 1                  | 0.801471  | 0.868566 |
|  1 | 2                  | 0.917279  | 0.85386  |
|  2 | 3                  | 0.684855  | 0.842722 |
|  3 | 4                  | 0.828715  | 0.831279 |
|  4 | 5                  | 0.842831  | 0.88511  |
|  5 | Max split          | 0.917279  | 0.88511  |
|  6 | Mean               | 0.81503   | 0.856308 |
|  7 | Standart deviation | 0.0755574 | 0.018957 |

#### Cross-val Top 1 accuracy

|    | Split              |      KNN |       SVM |
|---:|:-------------------|---------:|----------:|
|  0 | 1                  | 0.801471 | 0.868566  |
|  1 | 2                  | 0.917203 | 0.853726  |
|  2 | 3                  | 0.685373 | 0.840846  |
|  3 | 4                  | 0.828887 | 0.831647  |
|  4 | 5                  | 0.842686 | 0.885005  |
|  5 | Max split          | 0.917203 | 0.885005  |
|  6 | Mean               | 0.815124 | 0.855958  |
|  7 | Standart deviation | 0.075354 | 0.0191157 |

#### Cross-val Top 2 accuracy

|    | Split              |       KNN |        SVM |
|---:|:-------------------|----------:|-----------:|
|  0 | 1                  | 0.863051  | 1          |
|  1 | 2                  | 0.954922  | 1          |
|  2 | 3                  | 0.703772  | 1          |
|  3 | 4                  | 0.888684  | 1          |
|  4 | 5                  | 0.896044  | 0.9954     |
|  5 | Max split          | 0.954922  | 1          |
|  6 | Mean               | 0.861295  | 0.99908    |
|  7 | Standart deviation | 0.0843093 | 0.00183993 |

#### Cross-val Top 3 accuracy

|    | Split              |   KNN |        SVM |
|---:|:-------------------|------:|-----------:|
|  0 | 1                  |     1 | 1          |
|  1 | 2                  |     1 | 1          |
|  2 | 3                  |     1 | 1          |
|  3 | 4                  |     1 | 1          |
|  4 | 5                  |     1 | 0.99724    |
|  5 | Max split          |     1 | 1          |
|  6 | Mean               |     1 | 0.999448   |
|  7 | Standart deviation |     0 | 0.00110396 |

#### Test set basic

|    |     KNN |      SVM |
|---:|--------:|---------:|
|  0 | 0.94469 | 0.848451 |

#### Test set balanced

|    |     KNN |      SVM |
|---:|--------:|---------:|
|  0 | 0.94469 | 0.848451 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.94469  | 0.848451 |
|  1 | Top_2 | 0.975664 | 1        |
|  2 | Top_3 | 1        | 1        |

# 0001_2

## With level function

`$ python3 classification.py --n_jobs 16 cross-validation -db research_data/0001_2_308min/0001_2_308min_v3_vel10_acc2_clustered.joblib --output research_data/0001_2_308min/0001_2_308min_v3_vel10_acc2_clustered_2023-02.26.xlsx --classification_features_version v4 --stride 30 --param_set 4 --level`

Training dataset size: 9416                                                                               
Validation dataset size: 2838                                                                             

*Time: 434s*

#### Classifier parameters                                                                                
                                                                                                          
{'KNN': {'n_neighbors': 7}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 8000}}

#### Cross-val Basic accuracy

|    | Split              |      KNN |      SVM |                                                         
|---:|:-------------------|---------:|---------:|                                                         
|  0 | 1                  | 0.708068 | 0.728238 |                                                         
|  1 | 2                  | 0.579395 | 0.740308 |                                                         
|  2 | 3                  | 0.603824 | 0.618694 |                                                                                                                                                                   
|  3 | 4                  | 0.45778  | 0.457249 |                                                         
|  4 | 5                  | 0.698354 | 0.712693 |                                                         
|  5 | Max split          | 0.708068 | 0.740308 |                                                         
|  6 | Mean               | 0.609484 | 0.651436 |                                                         
|  7 | Standart deviation | 0.091179 | 0.106141 |                                                         
                                                                                                          
#### Cross-val Balanced accuracy                                                                          
                                                                                                          
|    | Split              |       KNN |      SVM |                                                        
|---:|:-------------------|----------:|---------:|                                                        
|  0 | 1                  | 0.708159  | 0.72964  |                                                        
|  1 | 2                  | 0.579757  | 0.739899 |                                                        
|  2 | 3                  | 0.603999  | 0.618668 |                                                        
|  3 | 4                  | 0.457537  | 0.458199 |                                                        
|  4 | 5                  | 0.698534  | 0.71392  |                                                        
|  5 | Max split          | 0.708159  | 0.739899 |                                                        
|  6 | Mean               | 0.609597  | 0.652065 |                                                        
|  7 | Standart deviation | 0.0912886 | 0.106074 |                                                        
                                                                                                          
#### Cross-val Top 1 accuracy                                                                             
                                                                                                          
|    | Split              |      KNN |      SVM |                                                         
|---:|:-------------------|---------:|---------:|                                                         
|  0 | 1                  | 0.708068 | 0.727176 |                                                         
|  1 | 2                  | 0.579395 | 0.740308 |                                                         
|  2 | 3                  | 0.603824 | 0.620287 |                                                         
|  3 | 4                  | 0.45778  | 0.458311 |                                                         
|  4 | 5                  | 0.698354 | 0.713755 |                                                         
|  5 | Max split          | 0.708068 | 0.740308 |                                                         
|  6 | Mean               | 0.609484 | 0.651967 |                                                         
|  7 | Standart deviation | 0.091179 | 0.105626 |                                                         
                                                                                                          
#### Cross-val Top 2 accuracy                                                                             
                                                                                                          
|    | Split              |       KNN |       SVM |                                                       
|---:|:-------------------|----------:|----------:|                                                       
|  0 | 1                  | 0.786624  | 0.888004  |                                                       
|  1 | 2                  | 0.689857  | 0.889007  |                                                       
|  2 | 3                  | 0.745088  | 0.883165  |                                                       
|  3 | 4                  | 0.57196   | 0.684015  |                                                       
|  4 | 5                  | 0.767392  | 0.847584  |                                                       
|  5 | Max split          | 0.786624  | 0.889007  |                                                       
|  6 | Mean               | 0.712184  | 0.838355  |                                                       
|  7 | Standart deviation | 0.0772442 | 0.0786698 |                                                       
                                                                                                          
#### Cross-val Top 3 accuracy                                                                             
                                                                                                          
|    | Split              |       KNN |       SVM |                                                       
|---:|:-------------------|----------:|----------:|                                                       
|  0 | 1                  | 0.806794  | 0.942675  |                                                       
|  1 | 2                  | 0.711099  | 0.958577  |                                                       
|  2 | 3                  | 0.79129   | 0.97026   |                                                       
|  3 | 4                  | 0.611259  | 0.813064  |                                                       
|  4 | 5                  | 0.797132  | 0.894849  |                                                       
|  5 | Max split          | 0.806794  | 0.97026   |                                                       
|  6 | Mean               | 0.743515  | 0.915885  |                                                       
|  7 | Standart deviation | 0.0744366 | 0.0574658 |                                                       
                                                                                                          
#### Test set basic                                                                                       
                                                                                                          
|    |      KNN |      SVM |                                                                              
|---:|---------:|---------:|                                                                              
|  0 | 0.636716 | 0.621212 |                                                                              
                                                                                                          
#### Test set balanced                                                                                    

|    |      KNN |      SVM |                                                                              
|---:|---------:|---------:|                                                                              
|  0 | 0.636716 | 0.621212 |                                                                              
                                                                                                          
#### Test set top k                                                                                       

|    | Top   |      KNN |      SVM |                                                                      
|---:|:------|---------:|---------:|                                                                      
|  0 | Top_1 | 0.636716 | 0.621212 |                                                                      
|  1 | Top_2 | 0.739958 | 0.968288 |                                                                      
|  2 | Top_3 | 0.761099 | 0.973573 |                                                                     

`$ python3 classification.py --n_jobs 16 cross-validation -db research_data/0001_2_308min/0001_2_308min_v3_vel10_acc2_clustered.joblib --output research_data/0001_2_308min/0001_2_308min_v3_vel10_acc2_clustered_2023-02.26.xlsx --classification_features_version v4 --stride 30 --param_set 1 --level`

Training dataset size: 9416                                                                               
Validation dataset size: 2838 

*Time: 432 s*                                                                                             

#### Classifier parameters                                                                                

{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 1000}}

#### Cross-val Basic accuracy                      

|    | Split              |       KNN |      SVM | 
|---:|:-------------------|----------:|---------:| 
|  0 | 1                  | 0.71603   | 0.739915 | 
|  1 | 2                  | 0.593202  | 0.734997 |
|  2 | 3                  | 0.595858  | 0.621349 | 
|  3 | 4                  | 0.453001  | 0.459373 |                                                        
|  4 | 5                  | 0.69145   | 0.699416 |
|  5 | Max split          | 0.71603   | 0.739915 |                                                        
|  6 | Mean               | 0.609908  | 0.65101  |
|  7 | Standart deviation | 0.0927441 | 0.104791 |                                           
                                                                                                          
#### Cross-val Balanced accuracy                   
                                                                                                                               
|    | Split              |       KNN |      SVM | 
|---:|:-------------------|----------:|---------:|                                                        
|  0 | 1                  | 0.716115  | 0.73971  | 
|  1 | 2                  | 0.593505  | 0.737284 |                                                        
|  2 | 3                  | 0.596065  | 0.62132  |
|  3 | 4                  | 0.452731  | 0.458199 |                                                        
|  4 | 5                  | 0.69166   | 0.702243 |
|  5 | Max split          | 0.716115  | 0.73971  |                                                        
|  6 | Mean               | 0.610015  | 0.651751 | 
|  7 | Standart deviation | 0.0928745 | 0.105821 |                                                        
                                                                                                          
#### Cross-val Top 1 accuracy                                                                             
                                                                                                          
|    | Split              |       KNN |      SVM |                                                        
|---:|:-------------------|----------:|---------:|                                                                                                                                                                                                                                                                                                                                                                                      
|  0 | 1                  | 0.71603   | 0.739915 |                                                                                                             
|  1 | 2                  | 0.593202  | 0.736059 |                                                                             
|  2 | 3                  | 0.595858  | 0.619756 |                                                                                                             
|  3 | 4                  | 0.453001  | 0.458311 |                                                                             
|  4 | 5                  | 0.69145   | 0.705258 |                                                                             
|  5 | Max split          | 0.71603   | 0.739915 |                                                                             
|  6 | Mean               | 0.609908  | 0.65186  |                                                                             
|  7 | Standart deviation | 0.0927441 | 0.106006 |                                                                             
                                                                                                                                                               
#### Cross-val Top 2 accuracy                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
                                                                                                                                                               
|    | Split              |       KNN |       SVM |                                                                                                            
|---:|:-------------------|----------:|----------:|                                                                                                            
|  0 | 1                  | 0.812633  | 0.883227  |                                                                                                            
|  1 | 2                  | 0.738715  | 0.898566  |                                                                                                            
|  2 | 3                  | 0.788635  | 0.861922  |                                                                                                            
|  3 | 4                  | 0.681891  | 0.682422  |                                                                                                                                                                 
|  4 | 5                  | 0.804036  | 0.848646  |                                                                                                            
|  5 | Max split          | 0.812633  | 0.898566  |                                                                                                                                                                 
|  6 | Mean               | 0.765182  | 0.834957  |                                                                                                            
|  7 | Standart deviation | 0.0488861 | 0.0781763 |                                                                                                            
                                                                                                                                                               
#### Cross-val Top 3 accuracy                                                                                                                                                                                       
                                                                                                                                                               
|    | Split              |       KNN |       SVM |                                                                                                                                                                 
|---:|:-------------------|----------:|----------:|                                                                                                            
|  0 | 1                  | 0.843949  | 0.942675  |                                                                                                            
|  1 | 2                  | 0.774296  | 0.959639  |                                                                                                            
|  2 | 3                  | 0.822623  | 0.961232  |                                                                                                                                                                 
|  3 | 4                  | 0.734997  | 0.807223  |                                                                                                            
|  4 | 5                  | 0.826872  | 0.896442  |                                                                                                                                                                 
|  5 | Max split          | 0.843949  | 0.961232  |                                                                                                            
|  6 | Mean               | 0.800548  | 0.913442  |                                                                                                            
|  7 | Standart deviation | 0.0401247 | 0.0580417 |                                                                                                            
                                                                                                                                                               
#### Test set basic                                                                                                                                            

|    |      KNN |      SVM |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
|---:|---------:|---------:|                                                                                                                                                                                        
|  0 | 0.645525 | 0.618041 |                                                                                                                                                                                        

#### Test set balanced                                                                                                                                                                                              

|    |      KNN |      SVM |                                                                                                                                                                                        
|---:|---------:|---------:|                                                                                                                                                                                        
|  0 | 0.645525 | 0.618041 |                                                                                                                                                                                        

#### Test set top k                                                                                                                                                                                                 

|    | Top   |      KNN |      SVM |                                                                                                                                                                                
|---:|:------|---------:|---------:|                                                                                                                                                                                
|  0 | Top_1 | 0.645525 | 0.618041 |                                                                                                                                                                                
|  1 | Top_2 | 0.861875 | 0.960888 |                                                                                                                                                                                
|  2 | Top_3 | 0.890416 | 0.975335 |    

### feature v5

python3 classification.py --n_jobs 12 cross-validation -db research_data/0001_2_308min/0001_2_308min_v3_vel10_acc2_clustered.joblib --output research_data/0001_2_308min/tables/0001_2_308min_features-v5_2023-03-05.xlsx --classification_features_version v5 --level

Training dataset size: 41685
Validation dataset size: 9672

*Time: 792 s*

#### Classifier parameters

{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True}}

#### Cross-val Basic accuracy

|    | Split              |       KNN |       SVM |
|---:|:-------------------|----------:|----------:|
|  0 | 1                  | 0.512654  | 0.584983  |
|  1 | 2                  | 0.516253  | 0.541682  |
|  2 | 3                  | 0.490584  | 0.528248  |
|  3 | 4                  | 0.566511  | 0.592059  |
|  4 | 5                  | 0.555955  | 0.594458  |
|  5 | Max split          | 0.566511  | 0.594458  |
|  6 | Mean               | 0.528392  | 0.568286  |
|  7 | Standart deviation | 0.0284156 | 0.0277122 |

#### Cross-val Balanced accuracy

|    | Split              |       KNN |       SVM |
|---:|:-------------------|----------:|----------:|
|  0 | 1                  | 0.529614  | 0.603955  |
|  1 | 2                  | 0.529873  | 0.548303  |
|  2 | 3                  | 0.49946   | 0.542894  |
|  3 | 4                  | 0.571496  | 0.598908  |
|  4 | 5                  | 0.567249  | 0.606985  |
|  5 | Max split          | 0.571496  | 0.606985  |
|  6 | Mean               | 0.539538  | 0.580209  |
|  7 | Standart deviation | 0.0267858 | 0.0284286 |

#### Cross-val Top 1 accuracy

|    | Split              |       KNN |       SVM |
|---:|:-------------------|----------:|----------:|
|  0 | 1                  | 0.512654  | 0.585342  |
|  1 | 2                  | 0.516253  | 0.541442  |
|  2 | 3                  | 0.490584  | 0.528487  |
|  3 | 4                  | 0.566511  | 0.59182   |
|  4 | 5                  | 0.555955  | 0.587022  |
|  5 | Max split          | 0.566511  | 0.59182   |
|  6 | Mean               | 0.528392  | 0.566823  |
|  7 | Standart deviation | 0.0284156 | 0.0264182 |

#### Cross-val Top 2 accuracy

|    | Split              |      KNN |       SVM |
|---:|:-------------------|---------:|----------:|
|  0 | 1                  | 0.706729 | 0.743673  |
|  1 | 2                  | 0.700252 | 0.792251  |
|  2 | 3                  | 0.699292 | 0.700612  |
|  3 | 4                  | 0.721363 | 0.782056  |
|  4 | 5                  | 0.738755 | 0.822598  |
|  5 | Max split          | 0.738755 | 0.822598  |
|  6 | Mean               | 0.713278 | 0.768238  |
|  7 | Standart deviation | 0.014984 | 0.0421885 |

#### Cross-val Top 3 accuracy

|    | Split              |       KNN |       SVM |
|---:|:-------------------|----------:|----------:|
|  0 | 1                  | 0.757107  | 0.870217  |
|  1 | 2                  | 0.759266  | 0.884131  |
|  2 | 3                  | 0.751589  | 0.824037  |
|  3 | 4                  | 0.759866  | 0.883651  |
|  4 | 5                  | 0.819839  | 0.898884  |
|  5 | Max split          | 0.819839  | 0.898884  |
|  6 | Mean               | 0.769533  | 0.872184  |
|  7 | Standart deviation | 0.0253218 | 0.0257259 |

#### Test set basic

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.586952 | 0.551799 |

#### Test set balanced

|    |     KNN |      SVM |
|---:|--------:|---------:|
|  0 | 0.60071 | 0.564747 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.586952 | 0.551799 |
|  1 | Top_2 | 0.763544 | 0.754963 |
|  2 | Top_3 | 0.833954 | 0.865902 |