# Bellevue NE8th Dataset

python3 -m sklearnex classification.py --n-jobs 4 exitpoint-metrics --dataset /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_NE8th_24h_v2/Preprocessed/ --test 0.5 --preprocessed --min-samples 400 --max-eps 0.15 --mse 
0.2 --models SVM KNN DT                                                                                                                                                                       
Intel(R) Extension for Scikit-learn* enabled (https://github.com/intel/scikit-learn-intelex)                                                                                                  
2343                                                                                                                                                                                          
103                                                                                                                                                                                           
2171                                                                                                                                                                                          
904                                                                                                                                                                                           
63                                                                                                                                                                                            
316                                                                                                                                                                                           
1802                                                                                                                                                                                          
2819      
2851                                          
466                    
206                                                                                                                                                                                           
2683                                
196                                                                                            
2289                                                                                                                                                                                          
87                                  
2754                                           
2590                                                                                           
1950                                           
2161                                         
457                                                                                                                                     
2205                                           
889                              
1275                                                                                           
Dataset loaded in 143 s                                                                        
Number of tracks: 33580                                                                                                                                                                       
Feature vectors.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 33580/33580 [00:00<00:00, 78301.19it/s]
Shape of feature vectors: (33580, 4)                                                           
Classes: [-1  0  1  2  3  4  5  6  7  8  9 10]                                                                                                                                                
Number of labeled trajectories after clustering: 32023                                                                                                                                        
Clustering done in 70 s                                                                                                                                                                       
Reduce labels: 32023it [00:00, 709542.13it/s]                                                                                                                                                 
Clustered exit centroids: [2 3 1 2 1 3 0 1 0 2 0]                                                                                                                                             
Exit points clusters: [0 1 2 3]                
Exit point clustering done in 0 s                                                                                                                                                                                                                                               
Train test split done in 0 s                                                                                                                                                                                                                                                    
Size of training set: 16011                                                                    
Size of testing set: 16012                                                                     
Feature vectors generated in 6 s                                                               
Classifier SVM trained in 273 s                                                                                                         
Classifier SVM evaluation based on original clusters: balanced accuracy: 0.765712, top-1: 0.864495, top-2: 0.965156, top-3: 0.989440                                                                                                                                                                                         
Classifier SVM evaluation based on exit point centroids: balanced accuracy: 0.931430, top-1: 0.929261, top-2: 0.988346, top-3: 0.999292                                                                                                                                                                                      
Classifier KNN trained in 11 s                                                                                         
Classifier KNN evaluation based on original clusters: balanced accuracy: 0.921398, top-1: 0.956911, top-2: 0.990061, top-3: 0.997556                                                                                                                                                                                         
Classifier KNN evaluation based on exit point centroids: balanced accuracy: 0.987316, top-1: 0.987445, top-2: 0.997909, top-3: 0.997931                                                                                                                                                                                      
Classifier DT trained in 2 s                                                                                                            
Classifier DT evaluation based on original clusters: balanced accuracy: 0.926769, top-1: 0.956097, top-2: 0.970463, top-3: 0.973980                                                                                                                                                                                          
Classifier DT evaluation based on exit point centroids: balanced accuracy: 0.970338, top-1: 0.969177, top-2: 0.973186, top-3: 0.973186                                                                                                                                                                                       

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |                                                                                                
|:----|---------:|---------:|---------:|--------------------:|                                                                                                
| SVM | 0.864495 | 0.965156 | 0.98944  |            0.765712 |                                                                                                
| KNN | 0.956911 | 0.990061 | 0.997556 |            0.921398 |
| DT  | 0.956097 | 0.970463 | 0.97398  |            0.926769 |                                                                          

|     |    Top-1 |    Top-2 |    Top-3 |   Balanced Accuracy |
|:----|---------:|---------:|---------:|--------------------:|                                                                                                
| SVM | 0.929261 | 0.988346 | 0.999292 |            0.93143  |                                                                                                
| KNN | 0.987445 | 0.997909 | 0.997931 |            0.987316 |                                                                                                
| DT  | 0.969177 | 0.973186 | 0.973186 |            0.970338 |                                                         