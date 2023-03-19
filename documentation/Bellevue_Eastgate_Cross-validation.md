# Bellevue Eastgate dataset cross-validation results

## Clustering metadat

- Method: OPTICS, min-samples = 50, max-eps-distance = 0.2, xi = 0.15, p = 2
- Fefature vector used: 4D

## Cross validation with version 7 feature vectors used for classification 

`python3 -m sklearnex classification.py --n_jobs 18 cross-validation -db research_data/Bellevue_Eastgate/Bellevue_Eastgate_train_v2_10-19-20_11-09-15_clustered_4D_threshold-07.joblib --output research_data/Bellevue_Eastgate/tables/Bellevue_Eastgate_v7_4D.xlsx --classification_features_version v7 --stride 15`  

Training dataset size: 51531                                                                                                                      
Validation dataset size: 17771  
Number of clusters: 12  
*Time: 882 s*  
                                                                               
#### Classifier parameters                                     
                                                                               
{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}
                                                               
#### Cross-val Basic accuracy                        
                                                               
|    | Split              |        KNN |       SVM |           
|---:|:-------------------|-----------:|----------:| 
|  0 | 1                  | 0.948385   | 0.938391  |           
|  1 | 2                  | 0.973122   | 0.970503  | 
|  2 | 3                  | 0.966815   | 0.963225  | 
|  3 | 4                  | 0.970309   | 0.966912  |           
|  4 | 5                  | 0.973802   | 0.969241  |                           
|  5 | Max split          | 0.973802   | 0.970503  |           
|  6 | Mean               | 0.966487   | 0.961655  |                           
|  7 | Standart deviation | 0.00938022 | 0.0118923 | 
                                                               
#### Cross-val Balanced accuracy                     
                                                                               
|    | Split              |       KNN |       SVM |            
|---:|:-------------------|----------:|----------:|                            
|  0 | 1                  | 0.899376  | 0.873268  |            
|  1 | 2                  | 0.950947  | 0.939681  |  
|  2 | 3                  | 0.937932  | 0.929356  |  
|  3 | 4                  | 0.941062  | 0.932275  |                                       
|  4 | 5                  | 0.951507  | 0.934802  |
|  5 | Max split          | 0.951507  | 0.939681  |                                       
|  6 | Mean               | 0.936165  | 0.921877  |
|  7 | Standart deviation | 0.0191545 | 0.0245394 |
                                                               
#### Cross-val Top 1 accuracy                                  
                                                               
|    | Split              |        KNN |      SVM |                                       
|---:|:-------------------|-----------:|---------:|                                                                            
|  0 | 1                  | 0.948385   | 0.938391 |                            
|  1 | 2                  | 0.973122   | 0.970503 |                            
|  2 | 3                  | 0.966815   | 0.963225 |                                       
|  3 | 4                  | 0.970309   | 0.966912 |                            
|  4 | 5                  | 0.973802   | 0.969144 |                                       
|  5 | Max split          | 0.973802   | 0.970503 |                            
|  6 | Mean               | 0.966487   | 0.961635 |                            
|  7 | Standart deviation | 0.00938022 | 0.01188  |                            
                                                                               
#### Cross-val Top 2 accuracy                                                  
                                                                                                          
|    | Split              |        KNN |      SVM |                                                                                                            
|---:|:-------------------|-----------:|---------:|                                       
|  0 | 1                  | 0.983021   | 0.968856 |                                       
|  1 | 2                  | 0.995343   | 0.994275 |                                                       
|  2 | 3                  | 0.997186   | 0.993402 |                                       
|  3 | 4                  | 0.996022   | 0.994372 |                                                       
|  4 | 5                  | 0.998059   | 0.994566 |                                       
|  5 | Max split          | 0.998059   | 0.994566 |                                       
|  6 | Mean               | 0.993926   | 0.989094 |                                       
|  7 | Standart deviation | 0.00553219 | 0.010127 |                                                       
                                                                                          
#### Cross-val Top 3 accuracy                                                                             
                                                                                          
|    | Split              |        KNN |        SVM |                                     
|---:|:-------------------|-----------:|-----------:|                                     
|  0 | 1                  | 0.985738   | 0.984283   |                                     
|  1 | 2                  | 0.999321   | 0.997865   |                                     
|  2 | 3                  | 0.999321   | 0.998933   |                                                     
|  3 | 4                  | 0.998545   | 0.998642   |                                                                                                                                
|  4 | 5                  | 0.999127   | 0.998448   |                                                                          
|  5 | Max split          | 0.999321   | 0.998933   |                                                     
|  6 | Mean               | 0.99641    | 0.995634   |                                                                          
|  7 | Standart deviation | 0.00534372 | 0.00568643 |                                                     
                                                                                                          
#### Test set basic                                                                                       

|    |      KNN |     SVM |                                                                               
|---:|---------:|--------:|                                                                                                    
|  0 | 0.973271 | 0.97299 |                                                                               
                                                                                                          
#### Test set balanced                                                                                    
                                                                                                          
|    |      KNN |      SVM |                                                                              
|---:|---------:|---------:|                                                                                                   
|  0 | 0.951146 | 0.945149 |                                                                                                                                                                                        

#### Test set top k                                                                                                            

|    | Top   |      KNN |      SVM |                                                                                           
|---:|:------|---------:|---------:|                                                                                           
|  0 | Top_1 | 0.973271 | 0.97299  |                                                                                           
|  1 | Top_2 | 0.997299 | 0.995836 |                                                                                           
|  2 | Top_3 | 0.999494 | 0.999156 |  

## Version 1

`python3 -m sklearnex classification.py --n_jobs 18 cross-validation -db research_data/Bellevue_Eastgate/Bellevue_Eastgate_train_v2_10-19-20_11-09-15_clustered_4D_threshold-07.joblib --output research_data/Bellevue_Eastgate/tables/Bellevue_Eastgate_v1_4D.xlsx --classification_features_version v1`  
Training dataset size: 17672  
Validation dataset size: 5888  
Number of clusters: 12  
*Time: 777 s*  

#### Classifier parameters

{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}

#### Cross-val Basic accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.964356   | 0.908345   |
|  1 | 2                  | 0.969165   | 0.910042   |
|  2 | 3                  | 0.973967   | 0.91313    |
|  3 | 4                  | 0.964629   | 0.908319   |
|  4 | 5                  | 0.965478   | 0.903226   |
|  5 | Max split          | 0.973967   | 0.91313    |
|  6 | Mean               | 0.967519   | 0.908612   |
|  7 | Standart deviation | 0.00365532 | 0.00321319 |

#### Cross-val Balanced accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.955655   | 0.853482   |
|  1 | 2                  | 0.954213   | 0.840585   |
|  2 | 3                  | 0.970227   | 0.856538   |
|  3 | 4                  | 0.954909   | 0.836173   |
|  4 | 5                  | 0.952573   | 0.839309   |
|  5 | Max split          | 0.970227   | 0.856538   |
|  6 | Mean               | 0.957515   | 0.845217   |
|  7 | Standart deviation | 0.00643712 | 0.00818101 |

#### Cross-val Top 1 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.964356   | 0.908062   |
|  1 | 2                  | 0.969165   | 0.909477   |
|  2 | 3                  | 0.973967   | 0.912847   |
|  3 | 4                  | 0.964629   | 0.90747    |
|  4 | 5                  | 0.965478   | 0.903226   |
|  5 | Max split          | 0.973967   | 0.912847   |
|  6 | Mean               | 0.967519   | 0.908216   |
|  7 | Standart deviation | 0.00365532 | 0.00311492 |

#### Cross-val Top 2 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.989816   | 0.975389   |
|  1 | 2                  | 0.995474   | 0.981612   |
|  2 | 3                  | 0.99519    | 0.981324   |
|  3 | 4                  | 0.990379   | 0.981324   |
|  4 | 5                  | 0.994624   | 0.982173   |
|  5 | Max split          | 0.995474   | 0.982173   |
|  6 | Mean               | 0.993096   | 0.980365   |
|  7 | Standart deviation | 0.00247021 | 0.00250707 |

#### Cross-val Top 3 accuracy

|    | Split              |        KNN |         SVM |
|---:|:-------------------|-----------:|------------:|
|  0 | 1                  | 0.996322   | 0.99604     |
|  1 | 2                  | 0.998303   | 0.997454    |
|  2 | 3                  | 0.998585   | 0.996321    |
|  3 | 4                  | 0.995473   | 0.99717     |
|  4 | 5                  | 0.998585   | 0.998302    |
|  5 | Max split          | 0.998585   | 0.998302    |
|  6 | Mean               | 0.997454   | 0.997058    |
|  7 | Standart deviation | 0.00130275 | 0.000812053 |

#### Test set basic

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.972486 | 0.918478 |

#### Test set balanced

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.965988 | 0.857468 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.972486 | 0.918478 |
|  1 | Top_2 | 0.995754 | 0.982846 |
|  2 | Top_3 | 0.999321 | 0.998981 |