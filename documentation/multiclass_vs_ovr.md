# Multiclass vs. OneVSRest Classifier results

## Multiclass

`python3 classification.py --n_jobs 16 cross-val-multiclass -db research_data/Bellevue_150th_Newport__2017-09/Bellevue_150th_Newport_train_v3_10-18-20_11-15-17.joblib --output research_data/Bellevue_150th_Newport__2017-09/tables/Bellevue_150th_Newport_features-v1_multiclass.xlsx --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2`
Training dataset size: 14341                                                                                                                                                                                                                                                                                                                                                                                                            
Validation dataset size: 4782                                                                                                                                                                                                                                                                                                                                                                                                           
Number of clusters: 11                                                                                                                                                                                                                                                                                                                                                     
*Time: 22 s*                                                                                                                                                                                                                                                                                                                                                               
#### Classifier parameters                                                                                                                                                                                                                                                                                                                                                 

{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}                                                                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                                                                           
#### Cross-val Basic accuracy                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                           
|    | Split              |        KNN |         SVM |                                                                                                                                                                                                                                                                                                                     
|---:|:-------------------|-----------:|------------:|                                                                                                                                                                                                                                                                                                                     
|  0 | 1                  | 0.944929   | 0.805856    |                                                                                                                                                                                                                                                                                                                     
|  1 | 2                  | 0.957113   | 0.807531    |                                                                                                                                                                                                                                                                                                                     
|  2 | 3                  | 0.949791   | 0.805439    |                                                                                                                                                                                                                                                                                                                     
|  3 | 4                  | 0.953975   | 0.806137    |                                                                                                                                                                                                                                                                                                                     
|  4 | 5                  | 0.959902   | 0.807183    |                                                                                                                                                                                                                                                                                                                     
|  5 | Max split          | 0.959902   | 0.807531    |                                                                                                                                                                                                                                                                                                                     
|  6 | Mean               | 0.953142   | 0.806429    |                                                                                                                                                                                                                                                                                                                     
|  7 | Standart deviation | 0.00530766 | 0.000797106 |                                                                                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                                                                                           
#### Cross-val Balanced accuracy                                                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                           
|    | Split              |       KNN |        SVM |                                                                                                                                                                                                                                                                                                                       
|---:|:-------------------|----------:|-----------:|                                                                                                                                                                                                                                                                                                                       
|  0 | 1                  | 0.89875   | 0.647221   |                                                                                                                                                                                                                                                                                                                       
|  1 | 2                  | 0.917943  | 0.639128   |                                                                                                                                                                                                                                                                                                                       
|  2 | 3                  | 0.896944  | 0.629471   |                                                                                                                                                                                                                                                                                                                       
|  3 | 4                  | 0.910926  | 0.631489   |                                                                                                                                                                                                                                                                                                                       
|  4 | 5                  | 0.924837  | 0.648215   |                                                                                                                                                                                                                                                                                                                       
|  5 | Max split          | 0.924837  | 0.648215   |                                                                                                                                                                                                                                                                                                                       
|  6 | Mean               | 0.90988   | 0.639105   |                                                                                                                                                                                                                                                                                                                       
|  7 | Standart deviation | 0.0107799 | 0.00774199 |                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                                                                                                                                                                           
#### Cross-val Top 1 accuracy                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                           
|    | Split              |       KNN |        SVM |                                                                                                                                                                                                                                                                                                                       
|---:|:-------------------|----------:|-----------:|                                                                                                                                                                                                                                                                                                                       
|  0 | 1                  | 0.944929  | 0.921924   |                                                                                                                                                                                                                                                                                                                       
|  1 | 2                  | 0.957113  | 0.918759   |                                                                                                                                                                                                                                                                                                                       
|  2 | 3                  | 0.950488  | 0.926778   |                                                                                                                                                                                                                                                                                                                       
|  3 | 4                  | 0.954672  | 0.919107   |                                                                                                                                                                                                                                                                                                                       
|  4 | 5                  | 0.959902  | 0.936541   |                                                                                                                                                                                                                                                                                                                       
|  5 | Max split          | 0.959902  | 0.936541   |                                                                                                                                                                                                                                                                                                                       
|  6 | Mean               | 0.953421  | 0.924622   |                                                                                                                                                                                                                                                                                                                       
|  7 | Standart deviation | 0.0052522 | 0.00661488 |                                                                                                                                                                                                                                                                                                                       
                                                                                                                                                                                                                    
#### Cross-val Top 2 accuracy                                                                             
                                                     
|    | Split              |        KNN |        SVM |                                                                          
|---:|:-------------------|-----------:|-----------:|                                                     
|  0 | 1                  | 0.974556   | 0.981178   |                                                     
|  1 | 2                  | 0.979777   | 0.98152    |                                                     
|  2 | 3                  | 0.978033   | 0.984658   |                                                     
|  3 | 4                  | 0.979428   | 0.981869   |                                                                                                                                                               
|  4 | 5                  | 0.981869   | 0.988494   |                                                     
|  5 | Max split          | 0.981869   | 0.988494   |                                                     
|  6 | Mean               | 0.978733   | 0.983544   |                                                     
|  7 | Standart deviation | 0.00242268 | 0.00276552 |                                                     
                                                                                                          
#### Cross-val Top 3 accuracy                                                                             
                                                                                                          
|    | Split              |        KNN |        SVM |                                                     
|---:|:-------------------|-----------:|-----------:|                                                     
|  0 | 1                  | 0.976995   | 0.989195   |  
|  1 | 2                  | 0.985356   | 0.990237   |                                                                                                                                                               
|  2 | 3                  | 0.98152    | 0.99477    |  
|  3 | 4                  | 0.982218   | 0.992678   |                                                                                                                                                               
|  4 | 5                  | 0.985356   | 0.995119   |  
|  5 | Max split          | 0.985356   | 0.995119   |                           
|  6 | Mean               | 0.982289   | 0.9924     |  
|  7 | Standart deviation | 0.00307981 | 0.00236792 |  
                                                                                                          
#### Test set basic                                                                                       
                                                     
|    |      KNN |      SVM |                         
|---:|---------:|---------:|                                                                                                                                                                                        
|  0 | 0.956713 | 0.815558 |                        
                                                     
#### Test set balanced                              
                                                     
|    |      KNN |      SVM |                        
|---:|---------:|---------:|                        
|  0 | 0.923756 | 0.674158 |                        
                                                     
#### Test set top k                                 
                                                     
|    | Top   |      KNN |      SVM |                           
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.956504 | 0.925972 |                           
|  1 | Top_2 | 0.979716 | 0.985153 |                                                                      
|  2 | Top_3 | 0.982434 | 0.994354 | 

## OVR

`python3 classification.py --n_jobs 16 cross-validation -db research_data/Bellevue_150th_Newport__2017-09/Bellevue_150th_Newport_train_v3_10-18-20_11-15-17.joblib --output research_data/Bellevue_150th_Newport__2017-09/tables/Bellevue_150th_Newport_features-v1_2023.xlsx --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2`
                                                                                                                                                                                                                    
Training dataset size: 14341                                                                                                                                                                                        
Validation dataset size: 4782                                                                                                                                                                                       
                                                                                                          
Number of clusters: 11                                                                                    
                                                                                                          
*Time: 686 s*                                                                                                                                                                                                       
                                                                                                          
#### Classifier parameters                                                                                
                                                                                                                                                                                                                    
{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 26000}}                                                                                                                      
                                                                                                          
#### Cross-val Basic accuracy                                                                                                                                                                                       
                                                                                                          
|    | Split              |        KNN |        SVM |                                                     
|---:|:-------------------|-----------:|-----------:|                                                     
|  0 | 1                  | 0.944929   | 0.868247   |                                                     
|  1 | 2                  | 0.954324   | 0.870642   |                                                     
|  2 | 3                  | 0.945258   | 0.877266   |                                                     
|  3 | 4                  | 0.951534   | 0.881102   |                                                     
|  4 | 5                  | 0.956067   | 0.87099    |                                                     
|  5 | Max split          | 0.956067   | 0.881102   |                                                                                                                                                               
|  6 | Mean               | 0.950422   | 0.873649   |
|  7 | Standart deviation | 0.00458627 | 0.00477342 |

#### Cross-val Balanced accuracy                                                                          

|    | Split              |       KNN |        SVM |                                                      
|---:|:-------------------|----------:|-----------:|                                                      
|  0 | 1                  | 0.891465  | 0.750537   |                                                      
|  1 | 2                  | 0.904515  | 0.754854   |                                                      
|  2 | 3                  | 0.876796  | 0.758953   |                                                      
|  3 | 4                  | 0.89663   | 0.766965   |                                                      
|  4 | 5                  | 0.91      | 0.751248   |                                                      
|  5 | Max split          | 0.91      | 0.766965   |                                                      
|  6 | Mean               | 0.895881  | 0.756512   |                                                      
|  7 | Standart deviation | 0.0114732 | 0.00602306 |                                                      

#### Cross-val Top 1 accuracy                                                                             

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.944929   | 0.868247   |
|  1 | 2                  | 0.954324   | 0.870642   |
|  2 | 3                  | 0.945258   | 0.876569   |
|  3 | 4                  | 0.951534   | 0.879358   |
|  4 | 5                  | 0.956067   | 0.870642   |
|  5 | Max split          | 0.956067   | 0.879358   |
|  6 | Mean               | 0.950422   | 0.873091   |
|  7 | Standart deviation | 0.00458627 | 0.00416758 |

#### Cross-val Top 2 accuracy                                                                             

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.989543   | 0.951203   |
|  1 | 2                  | 0.98954    | 0.964435   |
|  2 | 3                  | 0.988145   | 0.962343   |
|  3 | 4                  | 0.992678   | 0.966179   |
|  4 | 5                  | 0.991283   | 0.961646   |
|  5 | Max split          | 0.992678   | 0.966179   |
|  6 | Mean               | 0.990238   | 0.961161   |
|  7 | Standart deviation | 0.00157452 | 0.00522876 |

#### Cross-val Top 3 accuracy                                                                             

|    | Split              |         KNN |         SVM |
|---:|:-------------------|------------:|------------:|
|  0 | 1                  | 0.991635    | 0.988149    |
|  1 | 2                  | 0.993724    | 0.98954     |
|  2 | 3                  | 0.993026    | 0.990586    |
|  3 | 4                  | 0.993724    | 0.98954     |
|  4 | 5                  | 0.994073    | 0.98954     |
|  5 | Max split          | 0.994073    | 0.990586    |
|  6 | Mean               | 0.993236    | 0.989471    |
|  7 | Standart deviation | 0.000869916 | 0.000775127 |

#### Test set basic                                                                                       

|    |     KNN |      SVM |                                                                               
|---:|--------:|---------:|                                                                               
|  0 | 0.95504 | 0.891886 |                                                                               

#### Test set balanced                                                                                    

|    |     KNN |      SVM |                                                                               
|---:|--------:|---------:|                                                                               
|  0 | 0.90567 | 0.776638 |                                                                               

#### Test set top k                                                                                       

|    | Top   |      KNN |      SVM |                                                                      
|---:|:------|---------:|---------:|                                                                      
|  0 | Top_1 | 0.95504  | 0.891886 |                                                                      
|  1 | Top_2 | 0.991217 | 0.965705 |                                                                      
|  2 | Top_3 | 0.993936 | 0.989753 |  

## Version 7 feature vectors

`python3 classification.py --n_jobs 16 cross-val-multiclass -db research_data/Bellevue_150th_Newport__2017-09/Bellevue_150th_Newport_train_v3_10-18-20_11-15-17.joblib --output research_data/Bellevue_150th_Newport__2017-09/tables/Bellevue_150th_Newport_features-v1_multiclass.xlsx --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2 --classification_features_version v7`
                                                                                                                                                                                                                                                                                                                                                                                                                                        
Training dataset size: 51203                                                                                                                                                                                                                                                                                                                                                                                                            
Validation dataset size: 15414                                                                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                        
Number of clusters: 11 

*Time: 80 s*                                           
                                                                                                          
#### Classifier parameters                             
                                                                                                          
{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}
                                                                                                          
#### Cross-val Basic accuracy                          
                                                                                                          
|    | Split              |        KNN |       SVM |                                                      
|---:|:-------------------|-----------:|----------:|                                                      
|  0 | 1                  | 0.946294   | 0.927253  |                                                      
|  1 | 2                  | 0.945513   | 0.938385  |                                                      
|  2 | 3                  | 0.93692    | 0.922859  |                                                      
|  3 | 4                  | 0.956543   | 0.94502   |                                                      
|  4 | 5                  | 0.957715   | 0.942187  |                                                      
|  5 | Max split          | 0.957715   | 0.94502   |                                                      
|  6 | Mean               | 0.948597   | 0.935141  |                                                      
|  7 | Standart deviation | 0.00771276 | 0.0086119 |                                                      
                                                                                                          
#### Cross-val Balanced accuracy                                                                          
                                                                                                          
|    | Split              |       KNN |       SVM |                                                       
|---:|:-------------------|----------:|----------:|                                                       
|  0 | 1                  | 0.898659  | 0.839772  |                                                       
|  1 | 2                  | 0.897853  | 0.861921  |                                                       
|  2 | 3                  | 0.856257  | 0.81181   |                                                       
|  3 | 4                  | 0.905513  | 0.867289  |                                                       
|  4 | 5                  | 0.911975  | 0.862783  |                                                       
|  5 | Max split          | 0.911975  | 0.867289  |                                                                                                                                                                                                                                                                                                                                                                                     
|  6 | Mean               | 0.894051  | 0.848715  |                                                       
|  7 | Standart deviation | 0.0195791 | 0.0207809 |                                                                                                                                                                 
                                                                                                                                                                                                                    
#### Cross-val Top 1 accuracy                                                                                                                                                                                       
                                                                                                          
|    | Split              |        KNN |        SVM |                                                     
|---:|:-------------------|-----------:|-----------:|                                                     
|  0 | 1                  | 0.945806   | 0.937799   |  
|  1 | 2                  | 0.946099   | 0.937897   |                                                     
|  2 | 3                  | 0.935846   | 0.942877   |                                                  
|  3 | 4                  | 0.956641   | 0.953711   |                                                     
|  4 | 5                  | 0.957617   | 0.951074   |                                               
|  5 | Max split          | 0.957617   | 0.953711   |                                                     
|  6 | Mean               | 0.948402   | 0.944672   |         
|  7 | Standart deviation | 0.00803103 | 0.00661901 |                                            
                                                                                                          
#### Cross-val Top 2 accuracy                                                                   
                                                                                                          
|    | Split              |        KNN |        SVM |                              
|---:|:-------------------|-----------:|-----------:|                  
|  0 | 1                  | 0.970608   | 0.983986   |                                               
|  1 | 2                  | 0.967093   | 0.988966   |                                                     
|  2 | 3                  | 0.959477   | 0.990431   |                                                                                                                                                               
|  3 | 4                  | 0.97832    | 0.991797   |                                                     
|  4 | 5                  | 0.977637   | 0.986914   |                                                                                                                                                               
|  5 | Max split          | 0.97832    | 0.991797   |                                                     
|  6 | Mean               | 0.970627   | 0.988419   |                           
|  7 | Standart deviation | 0.00700205 | 0.00274491 |                                                     
                                                                                                          
#### Cross-val Top 3 accuracy                                                                             
                                                                                                          
|    | Split              |        KNN |        SVM |                                                     
|---:|:-------------------|-----------:|-----------:|                                                     
|  0 | 1                  | 0.9751     | 0.990919   |
|  1 | 2                  | 0.976272   | 0.998047   |
|  2 | 3                  | 0.963675   | 0.998047   |
|  3 | 4                  | 0.981348   | 0.998535   |
|  4 | 5                  | 0.980762   | 0.996777   |
|  5 | Max split          | 0.981348   | 0.998535   |
|  6 | Mean               | 0.975431   | 0.996465   |
|  7 | Standart deviation | 0.00636292 | 0.00283368 |

#### Test set basic

|    |      KNN |     SVM |
|---:|---------:|--------:|
|  0 | 0.948878 | 0.92429 |

#### Test set balanced

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.901899 | 0.845924 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.948488 | 0.935254 |
|  1 | Top_2 | 0.972817 | 0.987414 |
|  2 | Top_3 | 0.977293 | 0.99734  |

## OVR

`python3 classification.py --n_jobs 16 cross-validation -db research_data/Bellevue_150th_Newport__2017-09/Bellevue_150th_Newport_train_v3_10-18-20_11-15-17.joblib --output research_data/Bellevue_150th_Newport__2017-09/tables/Bellevue_150th_Newport_features-v1_multiclass.xlsx --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2 --classification_features_version v7`

Training dataset size: 51203
Validation dataset size: 15414

Number of clusters: 11

*Time: 1003 s*

#### Classifier parameters

{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 26000}}

#### Cross-val Basic accuracy

|    | Split              |        KNN |       SVM |
|---:|:-------------------|-----------:|----------:|
|  0 | 1                  | 0.948443   | 0.920027  |
|  1 | 2                  | 0.947857   | 0.93321   |
|  2 | 3                  | 0.936041   | 0.908407  |
|  3 | 4                  | 0.957617   | 0.93457   |
|  4 | 5                  | 0.960547   | 0.937793  |
|  5 | Max split          | 0.960547   | 0.937793  |
|  6 | Mean               | 0.950101   | 0.926802  |
|  7 | Standart deviation | 0.00861475 | 0.0110133 |

#### Cross-val Balanced accuracy

|    | Split              |       KNN |       SVM |
|---:|:-------------------|----------:|----------:|
|  0 | 1                  | 0.89649   | 0.83815   |
|  1 | 2                  | 0.896951  | 0.865025  |
|  2 | 3                  | 0.847927  | 0.805657  |
|  3 | 4                  | 0.901807  | 0.858409  |
|  4 | 5                  | 0.910583  | 0.868751  |
|  5 | Max split          | 0.910583  | 0.868751  |
|  6 | Mean               | 0.890752  | 0.847198  |
|  7 | Standart deviation | 0.0220035 | 0.0233047 |

#### Cross-val Top 1 accuracy

|    | Split              |        KNN |       SVM |
|---:|:-------------------|-----------:|----------:|
|  0 | 1                  | 0.948443   | 0.920027  |
|  1 | 2                  | 0.947857   | 0.933112  |
|  2 | 3                  | 0.936041   | 0.908505  |
|  3 | 4                  | 0.957617   | 0.93457   |
|  4 | 5                  | 0.960547   | 0.937988  |
|  5 | Max split          | 0.960547   | 0.937988  |
|  6 | Mean               | 0.950101   | 0.926841  |
|  7 | Standart deviation | 0.00861475 | 0.0110088 |

#### Cross-val Top 2 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.987404   | 0.978225   |
|  1 | 2                  | 0.986525   | 0.978322   |
|  2 | 3                  | 0.977932   | 0.981838   |
|  3 | 4                  | 0.991406   | 0.98252    |
|  4 | 5                  | 0.989453   | 0.983691   |
|  5 | Max split          | 0.991406   | 0.983691   |
|  6 | Mean               | 0.986544   | 0.980919   |
|  7 | Standart deviation | 0.00462629 | 0.00224022 |

#### Cross-val Top 3 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.991505   | 0.989552   |
|  1 | 2                  | 0.991114   | 0.992774   |
|  2 | 3                  | 0.982131   | 0.994532   |
|  3 | 4                  | 0.994824   | 0.993066   |
|  4 | 5                  | 0.992188   | 0.995215   |
|  5 | Max split          | 0.994824   | 0.995215   |
|  6 | Mean               | 0.990352   | 0.993028   |
|  7 | Standart deviation | 0.00430976 | 0.00195926 |

#### Test set basic

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.952057 | 0.924809 |

#### Test set balanced

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.902845 | 0.854856 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.952057 | 0.924809 |
|  1 | Top_2 | 0.987479 | 0.979499 |
|  2 | Top_3 | 0.992409 | 0.991696 |
