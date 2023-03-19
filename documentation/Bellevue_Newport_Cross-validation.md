# Bellevue Newport dataset cross-validation results

## Clustering metadat

- Method: OPTICS, min-samples = 50, max-eps-distance = 0.1, xi = 0.15, p = 2
- Fefature vector used: 4D

## Cross validation with version 7 feature vectors used for classification 

`python3 -m sklearnex classification.py --n_jobs 10 cross-validation -db research_data/Bellevue_150th_Newport__2017-09/Bellevue_150th_Newport_train_v3_10-18-20_11-15-17_clustered_4D.joblib --output research_data/Bellevue_150th_Newport__2017-09/tables/Bellevue_150th_Newport__2017-09_v7.xlsx --stride 15 --classification_features_version v7`  
Training dataset size: 48391  
Validation dataset size: 17020  
Number of clusters: 11  
*Time: 894 s*  
                                                                               
#### Classifier parameters                                                                                                                                     
                                                                               
{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}                                                                 
                                                                               
#### Cross-val Basic accuracy                                                                                                                                  
                                                                               
|    | Split              |        KNN |        SVM |                                                                                                          
|---:|:-------------------|-----------:|-----------:|    
|  0 | 1                  | 0.92086    | 0.902573   |                                                                                                          
|  1 | 2                  | 0.899359   | 0.872804   |
|  2 | 3                  | 0.908969   | 0.89409    |                                                                                                          
|  3 | 4                  | 0.918578   | 0.884584   |
|  4 | 5                  | 0.923331   | 0.891403   |
|  5 | Max split          | 0.923331   | 0.902573   |                          
|  6 | Mean               | 0.914219   | 0.889091   |
|  7 | Standart deviation | 0.00888264 | 0.00997657 |                          
                                                                               
#### Cross-val Balanced accuracy                     
                                                                               
|    | Split              |       KNN |       SVM |      
|---:|:-------------------|----------:|----------:|                        
|  0 | 1                  | 0.906807  | 0.862985  |
|  1 | 2                  | 0.855429  | 0.800916  |                                                                                                            
|  2 | 3                  | 0.890201  | 0.845241  |                                                                                                            
|  3 | 4                  | 0.897494  | 0.835662  |                                                                                                            
|  4 | 5                  | 0.908366  | 0.844695  |                                                                                                            
|  5 | Max split          | 0.908366  | 0.862985  |
|  6 | Mean               | 0.891659  | 0.8379    |
|  7 | Standart deviation | 0.0192735 | 0.0205044 |                            
                                                                               
#### Cross-val Top 1 accuracy                        
                                                                               
|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.92086    | 0.902573   |
|  1 | 2                  | 0.899359   | 0.873011   |                  
|  2 | 3                  | 0.908969   | 0.89409    |
|  3 | 4                  | 0.918578   | 0.884894   |                  
|  4 | 5                  | 0.923331   | 0.891816   |
|  5 | Max split          | 0.923331   | 0.902573   |                          
|  6 | Mean               | 0.914219   | 0.889277   |
|  7 | Standart deviation | 0.00888264 | 0.00990133 |                          
                                                                               
#### Cross-val Top 2 accuracy                                                  
                                                                                                                                                               
|    | Split              |        KNN |        SVM |                          
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.989668   | 0.982023   |                          
|  1 | 2                  | 0.977991   | 0.974065   |
|  2 | 3                  | 0.981091   | 0.971895   |
|  3 | 4                  | 0.985328   | 0.974065   |
|  4 | 5                  | 0.990804   | 0.973445   |
|  5 | Max split          | 0.990804   | 0.982023   |                          
|  6 | Mean               | 0.984976   | 0.975099   |
|  7 | Standart deviation | 0.00489865 | 0.00355193 |                          
                                                                                                                                                               
#### Cross-val Top 3 accuracy                                                             
                                                                                          
|    | Split              |        KNN |        SVM |                                                     
|---:|:-------------------|-----------:|-----------:|                                     
|  0 | 1                  | 0.994834   | 0.994008   |                                                     
|  1 | 2                  | 0.982434   | 0.992147   |                                     
|  2 | 3                  | 0.984811   | 0.992457   |                                     
|  3 | 4                  | 0.990287   | 0.990287   |                                     
|  4 | 5                  | 0.99628    | 0.990494   |                                                     
|  5 | Max split          | 0.99628    | 0.994008   |                                     
|  6 | Mean               | 0.989729   | 0.991879   |                                                     
|  7 | Standart deviation | 0.00541649 | 0.00137036 |                                     
                                                                                          
#### Test set basic                                                                       
                                                                                          
|    |      KNN |      SVM |                                                              
|---:|---------:|---------:|                                                                              
|  0 | 0.914571 | 0.894947 |                                                                                                                                                         

#### Test set balanced                                                                                    

|    |      KNN |      SVM |                                                                              
|---:|---------:|---------:|                                                                              
|  0 | 0.895192 | 0.850372 |                                                                              

#### Test set top k                                                                                       

|    | Top   |      KNN |      SVM |                                                                      
|---:|:------|---------:|---------:|                                                                      
|  0 | Top_1 | 0.914571 | 0.894947 |                                                                      
|  1 | Top_2 | 0.98302  | 0.977262 |                                                                      
|  2 | Top_3 | 0.987544 | 0.991128 |                                                                      

## Version 1

`python3 -m sklearnex classification.py --n_jobs 10 cross-validation -db research_data/Bellevue_150th_Newport__2017-09/Bellevue_150th_Newport_train_v3_10-18-20_11-15-17_clustered_4D.joblib --output research_data/Bellevue_150th_Newport__2017-09/tables/Bellevue_150th_Newport__2017-09_v1_4D.xlsx --classification_features_version v1`   
Training dataset size: 14713  
Validation dataset size: 4885  
Number of clusters: 11  
*Time: 616 s*  

#### Classifier parameters

{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}

#### Cross-val Basic accuracy

|    | Split              |       KNN |       SVM |
|---:|:-------------------|----------:|----------:|
|  0 | 1                  | 0.940537  | 0.852192  |
|  1 | 2                  | 0.931363  | 0.856949  |
|  2 | 3                  | 0.936799  | 0.85491   |
|  3 | 4                  | 0.930659  | 0.837525  |
|  4 | 5                  | 0.926241  | 0.849762  |
|  5 | Max split          | 0.940537  | 0.856949  |
|  6 | Mean               | 0.93312   | 0.850268  |
|  7 | Standart deviation | 0.0050003 | 0.0068192 |

#### Cross-val Balanced accuracy

|    | Split              |        KNN |       SVM |
|---:|:-------------------|-----------:|----------:|
|  0 | 1                  | 0.905166   | 0.741828  |
|  1 | 2                  | 0.903144   | 0.746773  |
|  2 | 3                  | 0.91772    | 0.7473    |
|  3 | 4                  | 0.901107   | 0.710736  |
|  4 | 5                  | 0.888754   | 0.734524  |
|  5 | Max split          | 0.91772    | 0.7473    |
|  6 | Mean               | 0.903178   | 0.736232  |
|  7 | Standart deviation | 0.00924926 | 0.0135491 |

#### Cross-val Top 1 accuracy

|    | Split              |       KNN |        SVM |
|---:|:-------------------|----------:|-----------:|
|  0 | 1                  | 0.940537  | 0.852871   |
|  1 | 2                  | 0.931363  | 0.856949   |
|  2 | 3                  | 0.936799  | 0.855929   |
|  3 | 4                  | 0.930659  | 0.836166   |
|  4 | 5                  | 0.926241  | 0.849762   |
|  5 | Max split          | 0.940537  | 0.856949   |
|  6 | Mean               | 0.93312   | 0.850335   |
|  7 | Standart deviation | 0.0050003 | 0.00751748 |

#### Cross-val Top 2 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.997621   | 0.977234   |
|  1 | 2                  | 0.994224   | 0.976894   |
|  2 | 3                  | 0.993884   | 0.978593   |
|  3 | 4                  | 0.996261   | 0.968049   |
|  4 | 5                  | 0.994901   | 0.974847   |
|  5 | Max split          | 0.997621   | 0.978593   |
|  6 | Mean               | 0.995378   | 0.975124   |
|  7 | Standart deviation | 0.00138603 | 0.00373512 |

#### Cross-val Top 3 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.998641   | 0.997621   |
|  1 | 2                  | 0.995243   | 0.996942   |
|  2 | 3                  | 0.994563   | 0.997961   |
|  3 | 4                  | 0.996261   | 0.995241   |
|  4 | 5                  | 0.996261   | 0.994901   |
|  5 | Max split          | 0.998641   | 0.997961   |
|  6 | Mean               | 0.996194   | 0.996533   |
|  7 | Standart deviation | 0.00138271 | 0.00124277 |

#### Test set basic

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.932651 | 0.848106 |

#### Test set balanced

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.913681 | 0.739739 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.932651 | 0.848106 |
|  1 | Top_2 | 0.995496 | 0.977687 |
|  2 | Top_3 | 0.995906 | 0.997134 |