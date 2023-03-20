# Bellevue Eastgate dataset cross-validation results  

## Clustering metadat  

- Method: OPTICS, min-samples = 50, max-eps-distance = 0.2, xi = 0.15, p = 2
- Fefature vector used: 4D  
 
## Cross validation with version 7 feature vectors used for classification  

`python3 -m sklearnex classification.py --n_jobs 18 cross-validation -db research_data/Bellevue_NE8th/Bellevue_NE8th_train_v1_10-18_11-14_clustered_4D_threshold-07.joblib --output research_data/Bellevue_NE8th/tables/Bellevue_NE8th_v7_4D.xlsx --classification_features_version v7 --stride 15`  

Training dataset size: 5616  
Validation dataset size: 21248  

Number of clusters: 14  

*Time: 1258 s*  

#### Classifier parameters  
{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}  
#### Cross-val Basic accuracy  
|    | Split              |       KNN |       SVM |
|---:|:-------------------|----------:|----------:|
|  0 | 1                  | 0.973558  | 0.935185  |
|  1 | 2                  | 0.911503  | 0.890759  |
|  2 | 3                  | 0.948896  | 0.941684  |
|  3 | 4                  | 0.921563  | 0.923433  |
|  4 | 5                  | 0.922899  | 0.931713  |
|  5 | Max split          | 0.973558  | 0.941684  |
|  6 | Mean               | 0.935684  | 0.924555  |
|  7 | Standart deviation | 0.0226122 | 0.0178948 |
                                                   
#### Cross-val Balanced accuracy                   
                                                   
|    | Split              |       KNN |        SVM |
|---:|:-------------------|----------:|-----------:|
|  0 | 1                  | 0.955233  | 0.90824    |
|  1 | 2                  | 0.921545  | 0.900296   |
|  2 | 3                  | 0.946426  | 0.920746   |
|  3 | 4                  | 0.921924  | 0.914784   |
|  4 | 5                  | 0.923698  | 0.904745   |
|  5 | Max split          | 0.955233  | 0.920746   |
|  6 | Mean               | 0.933765  | 0.909762   |
|  7 | Standart deviation | 0.0142272 | 0.00725211 |
                                                   
#### Cross-val Top 1 accuracy                      
                                                   
|    | Split              |       KNN |       SVM |
|---:|:-------------------|----------:|----------:|
|  0 | 1                  | 0.973558  | 0.935185  |
|  1 | 2                  | 0.911503  | 0.890759  |
|  2 | 3                  | 0.948896  | 0.941774  |
|  3 | 4                  | 0.921563  | 0.923344  |
|  4 | 5                  | 0.922899  | 0.931802  |
|  5 | Max split          | 0.973558  | 0.941774  |
|  6 | Mean               | 0.935684  | 0.924573  |                            
|  7 | Standart deviation | 0.0226122 | 0.0179202 |                            
                                                                                                                                                               
#### Cross-val Top 2 accuracy                                                                                                                                  
                                                                               
|    | Split              |       KNN |         SVM |
|---:|:-------------------|----------:|------------:|                                                                                                          
|  0 | 1                  | 0.995816  | 0.992432    |
|  1 | 2                  | 0.941684  | 0.992521    |                                                                                                          
|  2 | 3                  | 0.977742  | 0.993768    |
|  3 | 4                  | 0.952012  | 0.991364    |                                                                                                                                
|  4 | 5                  | 0.959313  | 0.991987    |
|  5 | Max split          | 0.995816  | 0.993768    |                                                                                                                                                               
|  6 | Mean               | 0.965313  | 0.992415    |
|  7 | Standart deviation | 0.0192671 | 0.000791127 |                                                                                                                                
                                                   
#### Cross-val Top 3 accuracy                                                                                                                                                        
                                                                                          
|    | Split              |       KNN |         SVM |                                     
|---:|:-------------------|----------:|------------:|                                                                                                                                
|  0 | 1                  | 0.99724   | 0.996795    |                                                                                                                                
|  1 | 2                  | 0.944266  | 0.997062    |                                                                                                                                
|  2 | 3                  | 0.979078  | 0.998041    |                                                                                                                                
|  3 | 4                  | 0.952902  | 0.996439    |                                                                                                                                
|  4 | 5                  | 0.96047   | 0.997418    |                                                                                                                                
|  5 | Max split          | 0.99724   | 0.998041    |                                                                                                                                
|  6 | Mean               | 0.966791  | 0.997151    |                                     
|  7 | Standart deviation | 0.0190699 | 0.000548826 |                                                                                                                                
                                                                                                          
#### Test set basic                                                                                       

         
|    |      KNN |     SVM |                                                                               
|---:|---------:|--------:|                                                                               
|  0 | 0.929688 | 0.91458 |                                                                                                                                                                                         

         
#### Test set balanced                                                                                    
                                                                                                          
|    |      KNN |      SVM |                                                                              
|---:|---------:|---------:|                                                                                                                                                                                        
|  0 | 0.945232 | 0.925002 |                                                                              

         
#### Test set top k                                                                                                                                                                                                 

         
|    | Top   |      KNN |      SVM |                                                                      
|---:|:------|---------:|---------:|                                                                      
|  0 | Top_1 | 0.929688 | 0.91458  |                                                                      
|  1 | Top_2 | 0.970162 | 0.994964 |                                                                      
|  2 | Top_3 | 0.971056 | 0.998682 |                                                                      

## Version 1

`python3 -m sklearnex classification.py --n_jobs 18 cross-validation -db research_data/Bellevue_NE8th/Bellevue_NE8th_train_v1_10-18_11-14_clustered_4D_threshold-07.joblib --output research_data/Bellevue_NE8th/tables/Bellevue_NE8th_v1_4D.xlsx --classification_features_version v1`  

Training dataset size: 12175  
Validation dataset size: 4039  

Number of clusters: 14  
*Time: 857 s*  

#### Classifier parameters

{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}

#### Cross-val Basic accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.966735   | 0.89692    |
|  1 | 2                  | 0.959754   | 0.896509   |
|  2 | 3                  | 0.973306   | 0.901848   |
|  3 | 4                  | 0.968378   | 0.889938   |
|  4 | 5                  | 0.967146   | 0.897331   |
|  5 | Max split          | 0.973306   | 0.901848   |
|  6 | Mean               | 0.967064   | 0.896509   |
|  7 | Standart deviation | 0.00434154 | 0.00380847 |

#### Cross-val Balanced accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.958911   | 0.862277   |
|  1 | 2                  | 0.945477   | 0.855133   |
|  2 | 3                  | 0.965854   | 0.865093   |
|  3 | 4                  | 0.958686   | 0.85551    |
|  4 | 5                  | 0.958092   | 0.856441   |
|  5 | Max split          | 0.965854   | 0.865093   |
|  6 | Mean               | 0.957404   | 0.858891   |
|  7 | Standart deviation | 0.00660372 | 0.00403681 |

#### Cross-val Top 1 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.966735   | 0.897741   |
|  1 | 2                  | 0.959754   | 0.894867   |
|  2 | 3                  | 0.973306   | 0.901848   |
|  3 | 4                  | 0.968378   | 0.89076    |
|  4 | 5                  | 0.967146   | 0.897331   |
|  5 | Max split          | 0.973306   | 0.901848   |
|  6 | Mean               | 0.967064   | 0.896509   |
|  7 | Standart deviation | 0.00434154 | 0.00364556 |

#### Cross-val Top 2 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.994251   | 0.97577    |
|  1 | 2                  | 0.988912   | 0.977002   |
|  2 | 3                  | 0.99384    | 0.980698   |
|  3 | 4                  | 0.995483   | 0.972074   |
|  4 | 5                  | 0.997125   | 0.977413   |
|  5 | Max split          | 0.997125   | 0.980698   |
|  6 | Mean               | 0.993922   | 0.976591   |
|  7 | Standart deviation | 0.00275368 | 0.00278535 |

#### Cross-val Top 3 accuracy

|    | Split              |       KNN |        SVM |
|---:|:-------------------|----------:|-----------:|
|  0 | 1                  | 0.996304  | 0.993018   |
|  1 | 2                  | 0.990965  | 0.997536   |
|  2 | 3                  | 0.996304  | 0.997947   |
|  3 | 4                  | 0.996715  | 0.998768   |
|  4 | 5                  | 0.998357  | 0.997536   |
|  5 | Max split          | 0.998357  | 0.998768   |
|  6 | Mean               | 0.995729  | 0.996961   |
|  7 | Standart deviation | 0.0024994 | 0.00202194 |

#### Test set basic

|    |     KNN |      SVM |
|---:|--------:|---------:|
|  0 | 0.97128 | 0.911859 |

#### Test set balanced

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.966097 | 0.862654 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.97128  | 0.911859 |
|  1 | Top_2 | 0.994801 | 0.981183 |
|  2 | Top_3 | 0.995296 | 0.998267 |