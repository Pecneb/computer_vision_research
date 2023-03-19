# Bellevue Eastgate dataset cross-validation results

## Clustering metadat

- Method: OPTICS, min-samples = 50, max-eps-distance = 0.2, xi = 0.15, p = 2
- Fefature vector used: 4D

## Cross validation with version 7 feature vectors used for classification 

`python3 -m sklearnex classification.py --n_jobs 18 cross-validation -db research_data/Bellevue_150th_SE38th/Bellevue_150th_SE38th_train_v2_10-18-20_11-08-14_clustered_4D_threshold-07.joblib --output research_data/Bellevue_150th_SE38th/tables/Bellevue_SE_v7_clustered_4D_threshold-07.xlsx --classification_features_version v7 --stride 15`  
                                                                                                                                                                                     
Training dataset size: 45388  

Validation dataset size: 14583  

Number of clusters: 10  
*Time: 800 s*  
                                                                               
#### Classifier parameters                                                                                                                                                                                                                                                                                                    
                                                                               
{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}                                                                                                                                                                                                                                
                                                                               
#### Cross-val Basic accuracy                                                                                                                                                                                                                                                                                                 
                                                                               
|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|                  
|  0 | 1                  | 0.974444   | 0.955827   |
|  1 | 2                  | 0.970588   | 0.959242   |                          
|  2 | 3                  | 0.975766   | 0.959022   |
|  3 | 4                  | 0.962873   | 0.944475   |                          
|  4 | 5                  | 0.968271   | 0.952297   |
|  5 | Max split          | 0.975766   | 0.959242   |                          
|  6 | Mean               | 0.970388   | 0.954173   |                                                                                                          
|  7 | Standart deviation | 0.00461214 | 0.00546833 |                          
                                                                               
#### Cross-val Balanced accuracy                                               
                                                                               
|    | Split              |        KNN |        SVM |                          
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.879402   | 0.841164   |                          
|  1 | 2                  | 0.86659    | 0.838937   |
|  2 | 3                  | 0.879935   | 0.843363   |
|  3 | 4                  | 0.865676   | 0.82344    |
|  4 | 5                  | 0.86764    | 0.834408   |                                                                                                          
|  5 | Max split          | 0.879935   | 0.843363   |
|  6 | Mean               | 0.871849   | 0.836263   |
|  7 | Standart deviation | 0.00641733 | 0.00706312 |                                     
                                                                               
#### Cross-val Top 1 accuracy                                                             
                                                                               
|    | Split              |        KNN |        SVM |                          
|---:|:-------------------|-----------:|-----------:| 
|  0 | 1                  | 0.974444   | 0.955827   |                          
|  1 | 2                  | 0.970588   | 0.959242   | 
|  2 | 3                  | 0.975766   | 0.959022   | 
|  3 | 4                  | 0.962873   | 0.944585   | 
|  4 | 5                  | 0.968271   | 0.952297   | 
|  5 | Max split          | 0.975766   | 0.959242   | 
|  6 | Mean               | 0.970388   | 0.954195   | 
|  7 | Standart deviation | 0.00461214 | 0.00542929 |                                     
                                                                               
#### Cross-val Top 2 accuracy                                                             
                                                                               
|    | Split              |        KNN |        SVM |                          
|---:|:-------------------|-----------:|-----------:|                                     
|  0 | 1                  | 0.994272   | 0.99284    |                                     
|  1 | 2                  | 0.992179   | 0.988544   |                                                     
|  2 | 3                  | 0.995373   | 0.990857   |                                     
|  3 | 4                  | 0.990526   | 0.992619   |                                                     
|  4 | 5                  | 0.989975   | 0.990966   |                                     
|  5 | Max split          | 0.995373   | 0.99284    |                                     
|  6 | Mean               | 0.992465   | 0.991165   |                                     
|  7 | Standart deviation | 0.00208658 | 0.00154426 |                                                     
                                                                                          
#### Cross-val Top 3 accuracy                                                                             
                                                                                          
|    | Split              |        KNN |         SVM |                                    
|---:|:-------------------|-----------:|------------:|                                    
|  0 | 1                  | 0.995924   | 0.998237    |                                    
|  1 | 2                  | 0.996475   | 0.997466    |                                    
|  2 | 3                  | 0.998127   | 0.998678    |                                                    
|  3 | 4                  | 0.992729   | 0.998678    |                                                                                                                               
|  4 | 5                  | 0.991958   | 0.998237    |                                                                         
|  5 | Max split          | 0.998127   | 0.998678    |                                                    
|  6 | Mean               | 0.995043   | 0.998259    |                                                                         
|  7 | Standart deviation | 0.00233301 | 0.000442798 |                                                    
                                                                                                          
#### Test set basic                                                                                       

|    |      KNN |     SVM |                                                                               
|---:|---------:|--------:|                                                                                                    
|  0 | 0.969416 | 0.95193 |                                                                               
                                                                                                          
#### Test set balanced                                                                                    
                                                                                                          
|    |     KNN |      SVM |                                                                               
|---:|--------:|---------:|                                                                                                    
|  0 | 0.87783 | 0.840501 |                                                                                                                                                                                         

#### Test set top k                                                                                                            

|    | Top   |      KNN |      SVM |                                                                                           
|---:|:------|---------:|---------:|                                                                                           
|  0 | Top_1 | 0.969416 | 0.95193  |                                                                                           
|  1 | Top_2 | 0.991086 | 0.987383 |                                                                                           
|  2 | Top_3 | 0.992663 | 0.994994 | 

## Version 1

`python3 -m sklearnex classification.py --n_jobs 18 cross-validation -db research_data/Bellevue_150th_SE38th/Bellevue_150th_SE38th_train_v2_10-18-20_11-08-14_clustered_4D_threshold-07.joblib --output research_data/Bellevue_150th_SE38th/tables/Bellevue_SE_v1_clustered_4D_threshold-07.xlsx --classification_features_version v1`  

Training dataset size: 9148  
Validation dataset size: 3060  

Number of clusters: 10  

*Time: 598 s*  

#### Classifier parameters

{'KNN': {'n_neighbors': 15}, 'SVM': {'kernel': 'rbf', 'probability': True, 'max_iter': 16000}}

#### Cross-val Basic accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.960656   | 0.914754   |
|  1 | 2                  | 0.968306   | 0.920765   |
|  2 | 3                  | 0.947541   | 0.911475   |
|  3 | 4                  | 0.960634   | 0.916348   |
|  4 | 5                  | 0.95626    | 0.90924    |
|  5 | Max split          | 0.968306   | 0.920765   |
|  6 | Mean               | 0.958679   | 0.914516   |
|  7 | Standart deviation | 0.00678698 | 0.00398857 |

#### Cross-val Balanced accuracy

|    | Split              |       KNN |        SVM |
|---:|:-------------------|----------:|-----------:|
|  0 | 1                  | 0.881877  | 0.760057   |
|  1 | 2                  | 0.913478  | 0.770662   |
|  2 | 3                  | 0.862273  | 0.763228   |
|  3 | 4                  | 0.906668  | 0.782721   |
|  4 | 5                  | 0.882582  | 0.761791   |
|  5 | Max split          | 0.913478  | 0.782721   |
|  6 | Mean               | 0.889376  | 0.767692   |
|  7 | Standart deviation | 0.0185306 | 0.00833951 |

#### Cross-val Top 1 accuracy

|    | Split              |        KNN |       SVM |
|---:|:-------------------|-----------:|----------:|
|  0 | 1                  | 0.960656   | 0.914754  |
|  1 | 2                  | 0.968306   | 0.919672  |
|  2 | 3                  | 0.947541   | 0.912568  |
|  3 | 4                  | 0.960634   | 0.916894  |
|  4 | 5                  | 0.95626    | 0.910334  |
|  5 | Max split          | 0.968306   | 0.919672  |
|  6 | Mean               | 0.958679   | 0.914845  |
|  7 | Standart deviation | 0.00678698 | 0.0032572 |

#### Cross-val Top 2 accuracy

|    | Split              |        KNN |        SVM |
|---:|:-------------------|-----------:|-----------:|
|  0 | 1                  | 0.996175   | 0.99071    |
|  1 | 2                  | 0.996175   | 0.987978   |
|  2 | 3                  | 0.991803   | 0.98306    |
|  3 | 4                  | 0.995626   | 0.990159   |
|  4 | 5                  | 0.993986   | 0.983598   |
|  5 | Max split          | 0.996175   | 0.99071    |
|  6 | Mean               | 0.994753   | 0.987101   |
|  7 | Standart deviation | 0.00167899 | 0.00321709 |

#### Cross-val Top 3 accuracy

|    | Split              |         KNN |        SVM |
|---:|:-------------------|------------:|-----------:|
|  0 | 1                  | 0.999454    | 0.998361   |
|  1 | 2                  | 0.998361    | 0.998361   |
|  2 | 3                  | 0.997268    | 0.995628   |
|  3 | 4                  | 0.997266    | 0.997266   |
|  4 | 5                  | 0.99672     | 0.995079   |
|  5 | Max split          | 0.999454    | 0.998361   |
|  6 | Mean               | 0.997814    | 0.996939   |
|  7 | Standart deviation | 0.000978084 | 0.00136569 |

#### Test set basic

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.966667 | 0.916667 |

#### Test set balanced

|    |      KNN |      SVM |
|---:|---------:|---------:|
|  0 | 0.926677 | 0.770997 |

#### Test set top k

|    | Top   |      KNN |      SVM |
|---:|:------|---------:|---------:|
|  0 | Top_1 | 0.966667 | 0.916667 |
|  1 | Top_2 | 0.995752 | 0.986601 |
|  2 | Top_3 | 0.997386 | 0.997386 |