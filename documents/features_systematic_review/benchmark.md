# Inference Benchmarking

## NE12

### KNN

python3 trajectorynet/benchmark.py --config config/benchmark_config_files/ne12th.yaml 
Config: {'debug': False, 'dataset': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/', 'model': ['KNN'], 'output': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/', 'feature_vector_version': ['ReVeRs'], 'min_samples': 100, 'xi': 0.05, 'max_eps': 0.15, 'mse': 0.2, 'n_jobs': 8, 'cross_validation': False, 'grid_search': False, 'fov_correction': True, 'video_path': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th/Bellevue_116th_NE12th__2017-09-11_00-08-29.mp4', 'google_map_image': '/media/pecneb/970evoplus/cv_research_video_dataset/Google_Maps_Pics/Bellevue_116th_NE12th_google_maps.png', 'distance': 254, 'max_stride': 30, 'window_length': 7, 'preprocessing': {'enter_exit_distance': {'switch': True, 'threshold': 0.4}, 'edge_distance': {'switch': True, 'threshold': 0.7}, 'detection_distance': {'switch': False, 'threshold': 0.01}, 'filling': False}}
Model loaded from /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/models/KNN_ReVeRs.joblib
Loading datasets: 15it [00:48,  2.81s/it]Error loading /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/Bellevue_116th_NE12th__2017-09-11_17-08-39.joblib: 
Loading datasets: 21it [00:57,  2.73s/it]
Dataset: (19316,)
Filter out edge detections.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3836/3836 [00:00<00:00, 259752.83it/s]
Dataset after preprocessing: (3836,)
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3836/3836 [00:00<00:00, 569071.20it/s]
Labels: (3836,)
X: (2229,), Y: (2229,)
Reduce labels: 2229it [00:00, 4512115.64it/s]
Feature vector version: ReVeRs
Features for classification.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2229/2229 [00:05<00:00, 390.40it/s]
Number of feature vectors: (25369, 6)
Model: /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/models/KNN_ReVeRs.joblib
Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    73   2781.3 MiB   2781.3 MiB           1   @profile
    74                                         def load_model_wrapper(model_path: Path) -> Tuple[bool, OneVsRestClassifierWrapper]:
    75   2802.8 MiB     21.6 MiB           1       return load_model(model_path)


Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    68   2802.8 MiB   2802.8 MiB           1   @profile
    69                                         def run_inference(model: OneVsRestClassifierWrapper, X: np.ndarray):
    70   2811.1 MiB      8.3 MiB           1       return model.predict_proba(X)


Time taken: 0.6523768539999963 seconds (process time)

### SVM

python3 trajectorynet/benchmark.py --config config/benchmark_config_files/ne12th.yaml 
Config: {'debug': False, 'dataset': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/', 'model': ['SVM'], 'output': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/', 'feature_vector_version': ['ReVeRs'], 'min_samples': 100, 'xi': 0.05, 'max_eps': 0.15, 'mse': 0.2, 'n_jobs': 8, 'cross_validation': False, 'grid_search': False, 'fov_correction': True, 'video_path': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th/Bellevue_116th_NE12th__2017-09-11_00-08-29.mp4', 'google_map_image': '/media/pecneb/970evoplus/cv_research_video_dataset/Google_Maps_Pics/Bellevue_116th_NE12th_google_maps.png', 'distance': 254, 'max_stride': 30, 'window_length': 7, 'preprocessing': {'enter_exit_distance': {'switch': True, 'threshold': 0.4}, 'edge_distance': {'switch': True, 'threshold': 0.7}, 'detection_distance': {'switch': False, 'threshold': 0.01}, 'filling': False}}
Model loaded from /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/models/SVM_ReVeRs.joblib
Loading datasets: 15it [00:47,  2.78s/it]Error loading /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/Bellevue_116th_NE12th__2017-09-11_17-08-39.joblib: 
Loading datasets: 21it [00:56,  2.67s/it]
Dataset: (19316,)
Filter out edge detections.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3836/3836 [00:00<00:00, 245230.84it/s]
Dataset after preprocessing: (3836,)
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3836/3836 [00:00<00:00, 517209.40it/s]
Labels: (3836,)
X: (2229,), Y: (2229,)
Reduce labels: 2229it [00:00, 2983121.77it/s]
Feature vector version: ReVeRs
Features for classification.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2229/2229 [00:05<00:00, 388.38it/s]
Number of feature vectors: (25369, 6)
Model: /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/models/SVM_ReVeRs.joblib
Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    73   2779.8 MiB   2779.8 MiB           1   @profile
    74                                         def load_model_wrapper(model_path: Path) -> Tuple[bool, OneVsRestClassifierWrapper]:
    75   2779.8 MiB      0.0 MiB           1       return load_model(model_path)


Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    68   2779.8 MiB   2779.8 MiB           1   @profile
    69                                         def run_inference(model: OneVsRestClassifierWrapper, X: np.ndarray):
    70   2784.8 MiB      5.0 MiB           1       return model.predict_proba(X)


Time taken: 3.927469934000001 seconds (process time)

### DT

python3 trajectorynet/benchmark.py --config config/benchmark_config_files/ne12th.yaml 
Config: {'debug': False, 'dataset': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/', 'model': ['DT'], 'output': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/', 'feature_vector_version': ['ReVeRs'], 'min_samples': 100, 'xi': 0.05, 'max_eps': 0.15, 'mse': 0.2, 'n_jobs': 8, 'cross_validation': False, 'grid_search': False, 'fov_correction': True, 'video_path': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th/Bellevue_116th_NE12th__2017-09-11_00-08-29.mp4', 'google_map_image': '/media/pecneb/970evoplus/cv_research_video_dataset/Google_Maps_Pics/Bellevue_116th_NE12th_google_maps.png', 'distance': 254, 'max_stride': 30, 'window_length': 7, 'preprocessing': {'enter_exit_distance': {'switch': True, 'threshold': 0.4}, 'edge_distance': {'switch': True, 'threshold': 0.7}, 'detection_distance': {'switch': False, 'threshold': 0.01}, 'filling': False}}
Model loaded from /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/models/DT_ReVeRs.joblib
Loading datasets: 15it [00:47,  2.79s/it]Error loading /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/Bellevue_116th_NE12th__2017-09-11_17-08-39.joblib: 
Loading datasets: 21it [00:56,  2.70s/it]
Dataset: (19316,)
Filter out edge detections.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3836/3836 [00:00<00:00, 246701.07it/s]
Dataset after preprocessing: (3836,)
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3836/3836 [00:00<00:00, 462590.21it/s]
Labels: (3836,)
X: (2229,), Y: (2229,)
Reduce labels: 2229it [00:00, 4596412.79it/s]
Feature vector version: ReVeRs
Features for classification.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2229/2229 [00:05<00:00, 388.66it/s]
Number of feature vectors: (25369, 6)
Model: /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/models/DT_ReVeRs.joblib
Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    73   2782.5 MiB   2782.5 MiB           1   @profile
    74                                         def load_model_wrapper(model_path: Path) -> Tuple[bool, OneVsRestClassifierWrapper]:
    75   2782.6 MiB      0.2 MiB           1       return load_model(model_path)


/home/pecneb/miniconda3/envs/computer-vision/lib/python3.8/site-packages/sklearn/multiclass.py:503: RuntimeWarning: invalid value encountered in divide
  Y /= np.sum(Y, axis=1)[:, np.newaxis]
Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    68   2782.6 MiB   2782.6 MiB           1   @profile
    69                                         def run_inference(model: OneVsRestClassifierWrapper, X: np.ndarray):
    70   2788.6 MiB      5.9 MiB           1       return model.predict_proba(X)


Time taken: 0.011966326000006688 seconds (process time)

### MLP

python3 trajectorynet/benchmark.py --config config/benchmark_config_files/ne12th.yaml 
Config: {'debug': False, 'dataset': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/', 'model': ['MLP'], 'output': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/', 'feature_vector_version': ['ReVeRs'], 'min_samples': 100, 'xi': 0.05, 'max_eps': 0.15, 'mse': 0.2, 'n_jobs': 8, 'cross_validation': False, 'grid_search': False, 'fov_correction': True, 'video_path': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th/Bellevue_116th_NE12th__2017-09-11_00-08-29.mp4', 'google_map_image': '/media/pecneb/970evoplus/cv_research_video_dataset/Google_Maps_Pics/Bellevue_116th_NE12th_google_maps.png', 'distance': 254, 'max_stride': 30, 'window_length': 7, 'preprocessing': {'enter_exit_distance': {'switch': True, 'threshold': 0.4}, 'edge_distance': {'switch': True, 'threshold': 0.7}, 'detection_distance': {'switch': False, 'threshold': 0.01}, 'filling': False}}
Model loaded from /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/models/MLP_ReVeRs.joblib
Loading datasets: 15it [00:47,  2.77s/it]Error loading /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/Bellevue_116th_NE12th__2017-09-11_17-08-39.joblib: 
Loading datasets: 21it [00:56,  2.67s/it]
Dataset: (19316,)
Filter out edge detections.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3836/3836 [00:00<00:00, 251860.46it/s]
Dataset after preprocessing: (3836,)
Feature vectors.: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3836/3836 [00:00<00:00, 576101.05it/s]
Labels: (3836,)
X: (2229,), Y: (2229,)
Reduce labels: 2229it [00:00, 4481832.99it/s]
Feature vector version: ReVeRs
Features for classification.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2229/2229 [00:05<00:00, 392.67it/s]
Number of feature vectors: (25369, 6)
Model: /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_116th_NE12th_24h/models/MLP_ReVeRs.joblib
Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    73   2779.9 MiB   2779.9 MiB           1   @profile
    74                                         def load_model_wrapper(model_path: Path) -> Tuple[bool, OneVsRestClassifierWrapper]:
    75   2786.3 MiB      6.4 MiB           1       return load_model(model_path)


Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    68   2786.3 MiB   2786.3 MiB           1   @profile
    69                                         def run_inference(model: OneVsRestClassifierWrapper, X: np.ndarray):
    70   2846.5 MiB     60.2 MiB           1       return model.predict_proba(X)


Time taken: 2.6484420719999946 seconds (process time)


## Newport

### KNN

python3 trajectorynet/benchmark.py --config config/benchmark_config_files/newport.yaml 
Config: {'debug': False, 'dataset': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2', 'model': ['KNN'], 'output': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2', 'feature_vector_version': ['ReVeRs'], 'min_samples': 200, 'xi': 0.05, 'max_eps': 0.1, 'mse': 0.2, 'n_jobs': 8, 'cross_validation': False, 'grid_search': False, 'fov_correction': True, 'video_path': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport/Bellevue_150th_Newport__2017-09-11_05-08-29.mp4', 'google_map_image': '/media/pecneb/970evoplus/cv_research_video_dataset/Google_Maps_Pics/Bellevue_150th_Newport_google_maps.png', 'distance': 300, 'max_stride': 30, 'window_length': 7, 'preprocessing': {'enter_exit_distance': {'switch': True, 'threshold': 0.4}, 'edge_distance': {'switch': True, 'threshold': 0.7}, 'detection_distance': {'switch': False, 'threshold': 0.01}, 'filling': False}}
Model loaded from /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/models/KNN_ReVeRs.joblib
Loading datasets: 24it [03:27,  8.65s/it]
Dataset: (46090,)
Filter out edge detections.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20626/20626 [00:00<00:00, 198168.20it/s]
Dataset after preprocessing: (20626,)
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20626/20626 [00:00<00:00, 306140.79it/s]
Labels: (20626,)
X: (17196,), Y: (17196,)
Reduce labels: 17196it [00:00, 4575604.36it/s]
Feature vector version: ReVeRs
Features for classification.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17196/17196 [00:30<00:00, 565.40it/s]
Number of feature vectors: (137364, 6)
Model: /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/models/KNN_ReVeRs.joblib
Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    73  10348.7 MiB  10348.7 MiB           1   @profile
    74                                         def load_model_wrapper(model_path: Path) -> Tuple[bool, OneVsRestClassifierWrapper]:
    75  10451.2 MiB    102.5 MiB           1       return load_model(model_path)


Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    68  10451.2 MiB  10451.2 MiB           1   @profile
    69                                         def run_inference(model: OneVsRestClassifierWrapper, X: np.ndarray):
    70  10494.0 MiB     42.8 MiB           1       return model.predict_proba(X)


Time taken: 6.095650976999991 seconds (process time)

### SVM

python3 trajectorynet/benchmark.py --config config/benchmark_config_files/newport.yaml 
Config: {'debug': False, 'dataset': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2', 'model': ['SVM'], 'output': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2', 'feature_vector_version': ['ReVeRs'], 'min_samples': 200, 'xi': 0.05, 'max_eps': 0.1, 'mse': 0.2, 'n_jobs': 8, 'cross_validation': False, 'grid_search': False, 'fov_correction': True, 'video_path': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport/Bellevue_150th_Newport__2017-09-11_05-08-29.mp4', 'google_map_image': '/media/pecneb/970evoplus/cv_research_video_dataset/Google_Maps_Pics/Bellevue_150th_Newport_google_maps.png', 'distance': 300, 'max_stride': 30, 'window_length': 7, 'preprocessing': {'enter_exit_distance': {'switch': True, 'threshold': 0.4}, 'edge_distance': {'switch': True, 'threshold': 0.7}, 'detection_distance': {'switch': False, 'threshold': 0.01}, 'filling': False}}
Model loaded from /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/models/SVM_ReVeRs.joblib
Loading datasets: 24it [03:17,  8.24s/it]
Dataset: (46090,)
Filter out edge detections.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20626/20626 [00:00<00:00, 209258.66it/s]
Dataset after preprocessing: (20626,)
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20626/20626 [00:00<00:00, 337292.10it/s]
Labels: (20626,)
X: (17196,), Y: (17196,)
Reduce labels: 17196it [00:00, 2654103.09it/s]
Feature vector version: ReVeRs
Features for classification.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17196/17196 [00:31<00:00, 537.89it/s]
Number of feature vectors: (137364, 6)
Model: /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/models/SVM_ReVeRs.joblib
Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    73  10352.8 MiB  10352.8 MiB           1   @profile
    74                                         def load_model_wrapper(model_path: Path) -> Tuple[bool, OneVsRestClassifierWrapper]:
    75  10352.8 MiB      0.0 MiB           1       return load_model(model_path)


Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    68  10352.8 MiB  10352.8 MiB           1   @profile
    69                                         def run_inference(model: OneVsRestClassifierWrapper, X: np.ndarray):
    70  10378.1 MiB     25.3 MiB           1       return model.predict_proba(X)


Time taken: 115.69838379599997 seconds (process time)

### DT

python3 trajectorynet/benchmark.py --config config/benchmark_config_files/newport.yaml 
Config: {'debug': False, 'dataset': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2', 'model': ['DT'], 'output': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2', 'feature_vector_version': ['ReVeRs'], 'min_samples': 200, 'xi': 0.05, 'max_eps': 0.1, 'mse': 0.2, 'n_jobs': 8, 'cross_validation': False, 'grid_search': False, 'fov_correction': True, 'video_path': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport/Bellevue_150th_Newport__2017-09-11_05-08-29.mp4', 'google_map_image': '/media/pecneb/970evoplus/cv_research_video_dataset/Google_Maps_Pics/Bellevue_150th_Newport_google_maps.png', 'distance': 300, 'max_stride': 30, 'window_length': 7, 'preprocessing': {'enter_exit_distance': {'switch': True, 'threshold': 0.4}, 'edge_distance': {'switch': True, 'threshold': 0.7}, 'detection_distance': {'switch': False, 'threshold': 0.01}, 'filling': False}}
Model loaded from /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/models/DT_ReVeRs.joblib
Loading datasets: 24it [03:41,  9.23s/it]
Dataset: (46090,)
Filter out edge detections.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20626/20626 [00:00<00:00, 187883.92it/s]
Dataset after preprocessing: (20626,)
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20626/20626 [00:00<00:00, 302318.33it/s]
Labels: (20626,)
X: (17196,), Y: (17196,)
Reduce labels: 17196it [00:00, 4347775.73it/s]
Feature vector version: ReVeRs
Features for classification.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17196/17196 [00:31<00:00, 543.02it/s]
Number of feature vectors: (137364, 6)
Model: /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/models/DT_ReVeRs.joblib
Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    73  10350.8 MiB  10350.8 MiB           1   @profile
    74                                         def load_model_wrapper(model_path: Path) -> Tuple[bool, OneVsRestClassifierWrapper]:
    75  10350.9 MiB      0.2 MiB           1       return load_model(model_path)


/home/pecneb/miniconda3/envs/computer-vision/lib/python3.8/site-packages/sklearn/multiclass.py:503: RuntimeWarning: invalid value encountered in divide
  Y /= np.sum(Y, axis=1)[:, np.newaxis]
Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    68  10350.9 MiB  10350.9 MiB           1   @profile
    69                                         def run_inference(model: OneVsRestClassifierWrapper, X: np.ndarray):
    70  10378.1 MiB     27.2 MiB           1       return model.predict_proba(X)

Time taken: 0.05138451700003088 seconds (process time)

### MLP

python3 trajectorynet/benchmark.py --config config/benchmark_config_files/newport.yaml 
Config: {'debug': False, 'dataset': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2', 'model': ['MLP'], 'output': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2', 'feature_vector_version': ['ReVeRs'], 'min_samples': 200, 'xi': 0.05, 'max_eps': 0.1, 'mse': 0.2, 'n_jobs': 8, 'cross_validation': False, 'grid_search': False, 'fov_correction': True, 'video_path': '/media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport/Bellevue_150th_Newport__2017-09-11_05-08-29.mp4', 'google_map_image': '/media/pecneb/970evoplus/cv_research_video_dataset/Google_Maps_Pics/Bellevue_150th_Newport_google_maps.png', 'distance': 300, 'max_stride': 30, 'window_length': 7, 'preprocessing': {'enter_exit_distance': {'switch': True, 'threshold': 0.4}, 'edge_distance': {'switch': True, 'threshold': 0.7}, 'detection_distance': {'switch': False, 'threshold': 0.01}, 'filling': False}}
Model loaded from /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/models/MLP_ReVeRs.joblib
Loading datasets: 24it [03:33,  8.91s/it]
Dataset: (46090,)
Filter out edge detections.: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20626/20626 [00:00<00:00, 200979.71it/s]
Dataset after preprocessing: (20626,)
Feature vectors.: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 20626/20626 [00:00<00:00, 297013.49it/s]
Labels: (20626,)
X: (17196,), Y: (17196,)
Reduce labels: 17196it [00:00, 4522526.43it/s]
Feature vector version: ReVeRs
Features for classification.: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 17196/17196 [00:31<00:00, 552.73it/s]
Number of feature vectors: (137364, 6)
Model: /media/pecneb/970evoplus/cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/models/MLP_ReVeRs.joblib
Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    73  10347.8 MiB  10347.8 MiB           1   @profile
    74                                         def load_model_wrapper(model_path: Path) -> Tuple[bool, OneVsRestClassifierWrapper]:
    75  10347.8 MiB      0.0 MiB           1       return load_model(model_path)


Filename: trajectorynet/benchmark.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    68  10347.8 MiB  10347.8 MiB           1   @profile
    69                                         def run_inference(model: OneVsRestClassifierWrapper, X: np.ndarray):
    70  10380.2 MiB     32.4 MiB           1       return model.predict_proba(X)


Time taken: 56.474254556999995 seconds (process time)