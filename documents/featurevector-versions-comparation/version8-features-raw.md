|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |                                                                                                
|:----|---------:|---------:|---------:|--------------------:|                                                                                                
| SVM | 0.910636 | 0.981793 | 0.995448 |            0.798054 |                                                                                                
| KNN | 0.948415 | 0.9956   | 0.998028 |            0.892796 |                                                                                                
| DT  | 0.913367 | 0.933697 | 0.942497 |            0.844374 |                                                                                                
                                                                                                                                                              
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |                                                                                                
|:----|---------:|---------:|---------:|--------------------:|                                                                                                
| SVM | 0.940525 | 1        | 1        |            0.937148 |                                                                                                
| KNN | 0.971325 | 0.999241 | 0.999241 |            0.968384 |                                  
| DT  | 0.935063 | 0.941284 | 0.941284 |            0.943183 |                                  
                                                                                                                                                              
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)                                                                  
Dataset loaded in 221 s                                                                                                                                       
Number of tracks: 32261                                                                                                                                       
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32261/32261 [00:00<00:00, 539956.03it/s]                                                               
Shape of feature vectors: (32261, 4)                                                                                                                                                                                                                                                                                         
Classes: [-1  0  1  2  3  4  5  6  7  8]                                                                                                                      
Number of labeled trajectories after clustering: 32052                                                                                                                                                                                                                                                                       
Clustering done in 532 s                                                                                                                                                                                                                                                                                                     
Reduce labels: 32052it [00:00, 4256184.13it/s]                                                                                                                
Clustered exit centroids: [0 3 1 1 2 0 3 2 0]                                                                                                                                                                                                                                                                                
Exit points clusters: [0 1 2 3]                                                                                                                                                                                                                                                                                              
Exit point clustering done in 0 s                                                                                                                             
Train test split done in 0 s                                                                                                                                  
Size of training set: 25641                                                                                                                                   
Size of testing set: 6411                                                                                                                                     
Features for classification.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25641/25641 [00:30<00:00, 839.31it/s]                                                               
Features for classification.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6411/6411 [00:01<00:00, 3837.64it/s]                                                               
Feature vectors generated in 32 s                                                                                                                             
Number of feature vectors in training set: 175299                                                                                                             
Number of feature vectors in testing set: 45074                                                                                                               
Classifier SVM trained in 112 s                                                                                                                               
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.914078, top-1: 0.920708, top-2: 0.994121, top-3: 0.997338                                                                                                                                                                                         
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.936767, top-1: 0.937592, top-2: 0.998025, top-3: 0.999068                                                                                                                                                                                      
Classifier KNN trained in 2 s                                                                                                                                 
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.916089, top-1: 0.919599, top-2: 0.966965, top-3: 0.967742                                                                                                                                                                                         
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.929850, top-1: 0.928384, top-2: 0.963771, top-3: 0.963771                                                                                                                                                                                      
Classifier DT trained in 0 s                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
Classifier DT evaluation based on original clusters: balanced accuracy: 0.900518, top-1: 0.902937, top-2: 0.923348, top-3: 0.925833                                                                                                                                                                                          
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.918583, top-1: 0.909105, top-2: 0.916515, top-3: 0.916515                                                                                                                                                                                       
|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |                                                                                                
|:----|---------:|---------:|---------:|--------------------:|                                                                                                
| SVM | 0.920708 | 0.994121 | 0.997338 |            0.914078 |                                                                                                
| KNN | 0.919599 | 0.966965 | 0.967742 |            0.916089 |                                                                                                
| DT  | 0.902937 | 0.923348 | 0.925833 |            0.900518 |                                                                                                

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |                                                                                                
|:----|---------:|---------:|---------:|--------------------:|                                                                                                
| SVM | 0.937592 | 0.998025 | 0.999068 |            0.936767 |                                                                                                
| KNN | 0.928384 | 0.963771 | 0.963771 |            0.92985  |                                                                                                
| DT  | 0.909105 | 0.916515 | 0.916515 |            0.918583 |                                                                                                

Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)                                                                  
Dataset loaded in 50 s                                                                                                                                        
Number of tracks: 42162                                                                                                                                       
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 42162/42162 [00:00<00:00, 620476.85it/s]                                                               
Shape of feature vectors: (42162, 4)                                                                                                                          
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10]                                                                                                                
Number of labeled trajectories after clustering: 39325                                                                                                        
Clustering done in 656 s                                                                                                                                      
Reduce labels: 39325it [00:00, 4534460.61it/s]                                                                                                                
Clustered exit centroids: [3 0 1 1 2 1 0 0 2 3 2]                                                                                                             
Exit points clusters: [0 1 2 3]                                                                                                                               
Exit point clustering done in 0 s                                                                                                                             
Train test split done in 0 s                                                                                                                                  
Size of training set: 31460                                                                                                                                   
Size of testing set: 7865                                                                                                                                     
Features for classification.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 31460/31460 [00:21<00:00, 1461.88it/s]                                                               
Features for classification.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7865/7865 [00:02<00:00, 2643.27it/s]                                                               
Feature vectors generated in 24 s                                                                                                                             
Number of feature vectors in training set: 148994                                                                                                             
Number of feature vectors in testing set: 37426                                                                                                               
Classifier SVM trained in 150 s                                                                                                                               
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.695642, top-1: 0.793379, top-2: 0.953001, top-3: 0.989766                                                                                                                                                                                         
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.843364, top-1: 0.831347, top-2: 0.965959, top-3: 0.997141                                                                                                                                                                                      
Classifier KNN trained in 2 s                                                                                                                                 
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.748111, top-1: 0.813392, top-2: 0.956528, top-3: 0.967135                                                                                                                                                                                         
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.821369, top-1: 0.824587, top-2: 0.960215, top-3: 0.968044                                                                                                                                                                                      
Classifier DT trained in 0 s                                                                                                                                  
Classifier DT evaluation based on original clusters: balanced accuracy: 0.700966, top-1: 0.756266, top-2: 0.787901, top-3: 0.798295                                                                                                                                                                                          
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.765015, top-1: 0.765484, top-2: 0.793406, top-3: 0.793566                                                                                                                                                                                       

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |                                                                                                
|:----|---------:|---------:|---------:|--------------------:|                                                                                                
| SVM | 0.793379 | 0.953001 | 0.989766 |            0.695642 |                                                                                                
| KNN | 0.813392 | 0.956528 | 0.967135 |            0.748111 |                                                                                                
| DT  | 0.756266 | 0.787901 | 0.798295 |            0.700966 |                                                                                                

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |                                                                                                
|:----|---------:|---------:|---------:|--------------------:|                                                                                                
| SVM | 0.831347 | 0.965959 | 0.997141 |            0.843364 |                                                                                                
| KNN | 0.824587 | 0.960215 | 0.968044 |            0.821369 |                                                                                                
| DT  | 0.765484 | 0.793406 | 0.793566 |            0.765015 |                                                                                                

Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)                                                                  
Dataset loaded in 206 s                                                                                                                                       
Number of tracks: 21921                                                                                                                                       
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 21921/21921 [00:00<00:00, 561157.05it/s]                                                               
Shape of feature vectors: (21921, 4)                                                                                                                          
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10 11 12]                                                                                                          
Number of labeled trajectories after clustering: 21469                                                                                                        
Clustering done in 352 s                                                                                                                                      
Reduce labels: 21469it [00:00, 4398569.39it/s]                                                                                                                
Clustered exit centroids: [0 1 0 4 3 2 2 1 2 3 4 3 1]                                                                                                         
Exit points clusters: [0 1 2 3 4]                                                                                                                             
Exit point clustering done in 0 s                                                                                                                             
Train test split done in 0 s                                                                                                                                  
Size of training set: 10734                                                                                                                                   
Size of testing set: 10735                                                                                                                                    
Features for classification.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10734/10734 [00:11<00:00, 952.19it/s]                                                               
Features for classification.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10735/10735 [00:11<00:00, 920.57it/s]                                                               
Feature vectors generated in 23 s                                                                                                                             
Number of feature vectors in training set: 106417                                                                                                             
Number of feature vectors in testing set: 107886                                                                                                              
Classifier SVM trained in 87 s                                                                                                                                
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.750374, top-1: 0.855106, top-2: 0.983751, top-3: 0.995319                                                                                                                                                                                         
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.848961, top-1: 0.869093, top-2: 0.989860, top-3: 0.998211                                                                                                                                                                                      
Classifier KNN trained in 8 s                                                                                                                                 
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.846088, top-1: 0.880828, top-2: 0.961320, top-3: 0.961839                                                                                                                                                                                         
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.901828, top-1: 0.891580, top-2: 0.963424, top-3: 0.963591                                                                                                                                                                                      
Classifier DT trained in 0 s                                                                                                                                  
Classifier DT evaluation based on original clusters: balanced accuracy: 0.813651, top-1: 0.833046, top-2: 0.854634, top-3: 0.855514                                                                                                                                                                                          
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.853031, top-1: 0.846171, top-2: 0.862429, top-3: 0.862447                                                                                                                                                                                       

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |                                                                                                
|:----|---------:|---------:|---------:|--------------------:|                                                                                                
| SVM | 0.855106 | 0.983751 | 0.995319 |            0.750374 |                                                                                                
| KNN | 0.880828 | 0.96132  | 0.961839 |            0.846088 |                                                                                                
| DT  | 0.833046 | 0.854634 | 0.855514 |            0.813651 |                                                                                                

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |                                                                                                
|:----|---------:|---------:|---------:|--------------------:|                                                                                                
| SVM | 0.869093 | 0.98986  | 0.998211 |            0.848961 |                                                                                                
| KNN | 0.89158  | 0.963424 | 0.963591 |            0.901828 |                                                                                                
| DT  | 0.846171 | 0.862429 | 0.862447 |            0.853031 |                                                                                                

Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)                                                                  
Dataset loaded in 80 s                                                                                                                                        
Number of tracks: 33580                                                                                                                                       

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.942298 | 0.987029 | 0.994422 |            0.812574 |
| KNN | 0.952797 | 0.986167 | 0.993064 |            0.884943 |
| DT  | 0.920623 | 0.940818 | 0.949681 |            0.869979 |

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|
| SVM | 0.963295 | 0.997756 | 0.999528 |            0.956216 |
| KNN | 0.970514 | 0.990095 | 0.990114 |            0.967491 |
| DT  | 0.939591 | 0.945491 | 0.945496 |            0.941532 |

