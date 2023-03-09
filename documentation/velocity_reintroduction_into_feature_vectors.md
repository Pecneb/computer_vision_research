# 0001_2

## features v6 v1 (v4, v5 type sliding window)

python3 -m sklearnex classification.py --n_jobs 10 cross-validation -db research_data/0001_2_308min/0001_2_308min_v3_vel10_acc2_clustered.joblib --output research_data/0001_2_308min/tables/0001_2_308min_v3_vel10_acc2_v5_cross_val_2023-03-08_level-4_weight-preset-2.xlsx --param_set 1 --classification_features_version v6 --stride 15 --level 4.0 --weights_preset 2                                                                                                                                                                                                       
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
Features for classification.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1520/1520 [07:52<00:00,  3.22it/s]                                                                                                                                                                                                                                                                                                                              
Features for classification.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 507/507 [00:39<00:00, 12.96it/s]                                                                                                                                                                                                                                                                                                                              
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
[  1766   3766 221294   3536  11044  11542   6611  19208   7089   1946                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
   9210  18890   2653]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:16<00:00,  1.29s/it]                                                                                                                                                                                                                                                                                                                              
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
[1766. 7064. 7064. 7064. 7064. 7064. 7064. 7064. 7064. 7064. 7064. 7064.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
 7064.]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
[  398  1614 72950   816  3441  6523  4291  3584  2942   518  2557  6255                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
  1092]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 13/13 [00:00<00:00, 16.89it/s]                                                                                                                                                                                                                                                                                                                              
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]                                                                                                                                                                                                                                                                                                                                                                                              
[ 398. 1592. 1592. 1592. 1592. 1592. 1592. 1592. 1592. 1592. 1592. 1592.                                                                                                                                                                                                                                                                                                                                                                
 1592.]                                                                                                                                                                                                                                                                                                                                                                                                                                 
                                                                                                                                                                                                                                                                                                                                                                                                                                        
Training dataset size: 69726                                                                                                                                                                                                                                                                                                                                                                                                            
Validation dataset size: 17152                                                                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                        
Cross validate models:  50%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▌                                                                                                                                 | 1/2 [02:22<02:22, 142.81s/it]/home/pecneb/anaconda3/envs/yolov7/lib/python3.10/site-packages/scipy/optimize/_linesearch.py:305: LineSearchWarning: The line search algorithm did not converge
  warn('The line search algorithm did not converge', LineSearchWarning)                                                                                                                                                                                                                                                                                                                                                                 
/home/pecneb/anaconda3/envs/yolov7/lib/python3.10/site-packages/scipy/optimize/_linesearch.py:305: LineSearchWarning: The line search algorithm did not converge                                                                                                                                                                                                                                                                        
  warn('The line search algorithm did not converge', LineSearchWarning)                                                                                                                                                                                                                                                                                                                                                                 
/home/pecneb/anaconda3/envs/yolov7/lib/python3.10/site-packages/scipy/optimize/_linesearch.py:305: LineSearchWarning: The line search algorithm did not converge                                                                                                                                                                                                                                                                        
  warn('The line search algorithm did not converge', LineSearchWarning)                                                                                                                                                                                                                                                                                                                                                                 
/home/pecneb/anaconda3/envs/yolov7/lib/python3.10/site-packages/scipy/optimize/_linesearch.py:305: LineSearchWarning: The line search algorithm did not converge                                                                                                                                                                                                                                                                        
  warn('The line search algorithm did not converge', LineSearchWarning)                                                                                                                                                                                                                                                                                                                                                                 
Cross validate models: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [22:34<00:00, 677.41s/it]                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                        
*Time: 1354 s*                                                                                                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                        
#### Classifier parameters                                                                                                                                                                                                                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                                                                                                                                                                                        
{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}                                                                                                                                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                                                                                                                                                        
#### Cross-val Basic accuracy                                                                                                                                                                                                                                                                                                                                                                                                           
                                                                                                                                                                                                                                                                                                                                                                                                                                        
|    | Split              |       KNN |      SVM |                                                                                                                                                                                                                                                                                                                                                                                      
|---:|:-------------------|----------:|---------:|                                                                                                                                                                                                                                                                                                                                                                                      
|  0 | 1                  | 0.52524   | 0.53492  |                                                                                                                                                                                                                                                                                                                                                                                      
|  1 | 2                  | 0.550018  | 0.547867 |                                                                                                                                                                                                                                                                                                                                                                                      
|  2 | 3                  | 0.576049  | 0.597849 |                                                                                                                                                                                                                                                                                                                                                                                      
|  3 | 4                  | 0.525923  | 0.532951 |                                                                                                                                                                                                                                                                                                                                                                                      
|  4 | 5                  | 0.557978  | 0.554607 |                                                                                                                                                                                                                                                                                                                                                                                      
|  5 | Max split          | 0.576049  | 0.597849 |                                                        
|  6 | Mean               | 0.547042  | 0.553639 |                                                        
|  7 | Standart deviation | 0.0194482 | 0.023525 |                                                        
                                                                                                                                                                                                                                                                                                                                                                                                                                        
#### Cross-val Balanced accuracy                                                                                                                                                                                    
                                                                                                                                                                                                                    
|    | Split              |       KNN |       SVM |                                                                                                                                                                                                                                                                                                                                                                                     
|---:|:-------------------|----------:|----------:|                                                                                                                                                                                                                                                                                                                                                                                     
|  0 | 1                  | 0.550496  | 0.552809  |                                                       
|  1 | 2                  | 0.569525  | 0.558847  |                                                       
|  2 | 3                  | 0.571147  | 0.578933  |                                                       
|  3 | 4                  | 0.530367  | 0.526156  |                                                       
|  4 | 5                  | 0.577107  | 0.562032  |                                                                                                                                                                                                                                                                                                                                                                                     
|  5 | Max split          | 0.577107  | 0.578933  |                                                       
|  6 | Mean               | 0.559728  | 0.555755  |                                                       
|  7 | Standart deviation | 0.0171794 | 0.0171519 |                                                                                                                                                                                                                                                                                                                                                                                     
                                                                                                          
#### Cross-val Top 1 accuracy                                                                                                                                                                                                                                                                                                                                                                                                           
                                                                                                          
|    | Split              |       KNN |       SVM |                                                       
|---:|:-------------------|----------:|----------:|                                                       
|  0 | 1                  | 0.52524   | 0.53492   |                                                       
|  1 | 2                  | 0.550018  | 0.546289  |                                                       
|  2 | 3                  | 0.576049  | 0.597849  |                                                                                                                                                                 
|  3 | 4                  | 0.525923  | 0.533166  |                                                       
|  4 | 5                  | 0.557978  | 0.553962  |                                                       
|  5 | Max split          | 0.576049  | 0.597849  |                                                       
|  6 | Mean               | 0.547042  | 0.553237  |                                                       
|  7 | Standart deviation | 0.0194482 | 0.0235684 |                                                       
                                                                                                          
#### Cross-val Top 2 accuracy                                                                             
                                                                                                          
|    | Split              |       KNN |      SVM |                                                        
|---:|:-------------------|----------:|---------:|                                                        
|  0 | 1                  | 0.722931  | 0.731823 |                                                        
|  1 | 2                  | 0.768806  | 0.757189 |                                                        
|  2 | 3                  | 0.802653  | 0.786088 |                                                        
|  3 | 4                  | 0.75848   | 0.76142  |                                                        
|  4 | 5                  | 0.789889  | 0.783865 |                                                        
|  5 | Max split          | 0.802653  | 0.786088 |                                                        
|  6 | Mean               | 0.768552  | 0.764077 |                                                        
|  7 | Standart deviation | 0.0275714 | 0.019854 |                                                        
                                                                                                          
#### Cross-val Top 3 accuracy                                                                             
                                                                                                          
|    | Split              |       KNN |       SVM |                                                       
|---:|:-------------------|----------:|----------:|                                                       
|  0 | 1                  | 0.813997  | 0.841603  |                                                       
|  1 | 2                  | 0.860954  | 0.868627  |                                                       
|  2 | 3                  | 0.895446  | 0.88175   |                                                       
|  3 | 4                  | 0.853066  | 0.882395  |                                                       
|  4 | 5                  | 0.900896  | 0.899534  |                                                       
|  5 | Max split          | 0.900896  | 0.899534  |                                                       
|  6 | Mean               | 0.864872  | 0.874782  |                                                       
|  7 | Standart deviation | 0.0315454 | 0.0192763 |                                                       
                                                                                                          
#### Test set basic                                                                                       
                                                                                                          
|    |      KNN |      SVM |                                                                              
|---:|---------:|---------:|                                                                              
|  0 | 0.603603 | 0.585879 |                                                                              
                                                                                                          
#### Test set balanced                                                                                    
                                                                                                          
|    |      KNN |      SVM |                                                                              
|---:|---------:|---------:|                                                                              
|  0 | 0.603062 | 0.590392 |                                                                              
                                                                                                          
#### Test set top k                                                                                       
                                                                                                          
|    | Top   |      KNN |      SVM |                                                                      
|---:|:------|---------:|---------:|                                                                      
|  0 | Top_1 | 0.603603 | 0.585879 |                                                                      
|  1 | Top_2 | 0.803405 | 0.749767 |                                                                      
|  2 | Top_3 | 0.896455 | 0.864156 |                                                                      

## features v6 (v1 type sliding window)

python3 -m sklearnex classification.py --n_jobs 10 cross-validation -db research_data/0001_2_308min/0001_2_308min_v3_vel10_acc2_clustered.joblib --output research_data/0001_2_308min/tables/0001_2_308min_v3_vel10_acc2_v5_cross_val_2023-03-08_level-4_weight-preset-2.xlsx --param_set 1 --classification_features_version v6 --stride 15 --weights_preset 2                                                                                                                                                                                                                  

*Time: 402 s*

#### Classifier parameters

{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}

#### Cross-val Basic accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.841002   | 0.854214   |
|  1 | 2                  | 0.85877    | 0.853303   |
|  2 | 3                  | 0.854442   | 0.864692   |
|  3 | 4                  | 0.848713   | 0.855776   |
|  4 | 5                  | 0.852814   | 0.859421   |
|  5 | Max split          | 0.85877    | 0.864692   |
|  6 | Mean               | 0.851148   | 0.857481   |
|  7 | Standart deviation | 0.00600975 | 0.00416669 |

#### Cross-val Balanced accuracy

|    | Split              |       KNN |       SVM |
|---:|:-------------------|----------:|----------:|
|  0 | 1                  | 0.45142   | 0.464007  |
|  1 | 2                  | 0.490531  | 0.443896  |
|  2 | 3                  | 0.45167   | 0.45296   |
|  3 | 4                  | 0.446748  | 0.431161  |
|  4 | 5                  | 0.465221  | 0.452905  |
|  5 | Max split          | 0.490531  | 0.464007  |
|  6 | Mean               | 0.461118  | 0.448986  |
|  7 | Standart deviation | 0.0159486 | 0.0109581 |

#### Cross-val Top 1 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.841002   | 0.853759   |
|  1 | 2                  | 0.85877    | 0.853531   |
|  2 | 3                  | 0.854442   | 0.86492    |
|  3 | 4                  | 0.848713   | 0.856004   |
|  4 | 5                  | 0.852814   | 0.859877   |
|  5 | Max split          | 0.85877    | 0.86492    |
|  6 | Mean               | 0.851148   | 0.857618   |
|  7 | Standart deviation | 0.00600975 | 0.00430426 |

#### Cross-val Top 2 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.925513   | 0.918223   |
|  1 | 2                  | 0.945786   | 0.942141   |
|  2 | 3                  | 0.940547   | 0.930979   |
|  3 | 4                  | 0.934609   | 0.931875   |
|  4 | 5                  | 0.946913   | 0.930736   |
|  5 | Max split          | 0.946913   | 0.942141   |
|  6 | Mean               | 0.938673   | 0.930791   |
|  7 | Standart deviation | 0.00789568 | 0.00758934 |

#### Cross-val Top 3 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.94533    | 0.957631   |
|  1 | 2                  | 0.970843   | 0.967882   |
|  2 | 3                  | 0.961959   | 0.955125   |
|  3 | 4                  | 0.95671    | 0.959672   |
|  4 | 5                  | 0.967191   | 0.960811   |
|  5 | Max split          | 0.970843   | 0.967882   |
|  6 | Mean               | 0.960407   | 0.960224   |
|  7 | Standart deviation | 0.00892534 | 0.00428971 |

#### Test set basic

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.827656 | 0.820871 |

#### Test set balanced

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.469745 | 0.452543 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.827656 | 0.820871 |
|  1 | Top_2 | 0.92102  | 0.918306 |
|  2 | Top_3 | 0.955354 | 0.952232 |