User manual
===========

Create database from video files
--------------------------------

.. code-block:: console

    $ python3 trajectorynet/detect.py --help
        usage: detect.py [-h] --video VIDEO --outdir OUTDIR [--database] [--joblib] --yolo-model YOLO_MODEL [--iou IOU] [--score SCORE] [--device DEVICE] [--half] [--show] [--max-age MAX_AGE] [--model MODEL]

        Detect objects in a video.

        optional arguments:
        -h, --help            show this help message and exit
        --video VIDEO         Path to video file.
        --outdir OUTDIR       Path to output video file.
        --database            Save results to database.
        --joblib              Save results to database.
        --yolo-model YOLO_MODEL
                                Path to model weights file.
        --iou IOU             IoU threshold.
        --score SCORE         Score threshold.
        --device DEVICE       Device to run inference on.
        --half                Use half precision.
        --show                View output video.
        --max-age MAX_AGE     Max age of a track.
        --model MODEL         Path to trajectorynet model.



With this example command, the program will create a joblib file containing trajectory objects for each video file in the directory ``cv_research_video_dataset/Bellevue_116th_NE12th/``. The joblib file will be saved in the directory ``research_data/Bellevue_NE116th/``.

.. code-block:: console 

    $ python3 trajectorynet/detect.py --video ../../cv_research_video_dataset/Bellevue_116th_NE12th/Bellevue_116th_NE12th__2017-09-11_12-08-33.mp4 --outdir research_data/Bellevue_NE116th_test/ --joblib --iou 0.5 --device 0 --half --view --score 0.6 --model trajectorynet/yolov7/yolov7.pt


Run clustering on database
--------------------------

.. code-block:: console

    $ python3 trajectorynet/clustering.py -db research_data/Bellevue_NE116th_test/Bellevue_116th_NE12th.joblib --outdir research_data/Bellevue_NE116th_test/ --n-jobs 6 --dimensions 4D optics --min-samples 10 --max-eps 0.1 --xi 0.05

Train model
-----------

.. code-block:: console

    $ python3 trajectorynet/train.py --help
        usage: train.py [-h] --dataset DATASET --model {SVM,KNN,DT} --output OUTPUT [--feature-vector-version {1,7}] [--min-samples MIN_SAMPLES] [--xi XI] [--max-eps MAX_EPS] [--mse MSE] [--n-jobs N_JOBS] [--cross-validation]

        Train a model

        optional arguments:
        -h, --help            show this help message and exit
        --dataset DATASET     Path to dataset
        --model {SVM,KNN,DT}  Model to train
        --output OUTPUT       Path to save model
        --feature-vector-version {1,7}
                                Version of feature vector to use.
        --min-samples MIN_SAMPLES
                                Minimum number of samples in a cluster.
        --xi XI               Minimum difference between reachability distances.
        --max-eps MAX_EPS     Maximum reachability distance.
        --mse MSE             Mean square error.
        --n-jobs N_JOBS       Number of jobs to run in parallel.
        --cross-validation    Run cross validation.
 

    $ python3 trajectorynet/train.py --dataset ../../cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/Preprocessed_enter-exit-distance-0.7_v2/ --model SVM --feature-vector-version 7 --output ../../cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/Preprocessed_enter-exit-distance-0.7_v2/ --min-samples 200 --max-eps 0.1 --mse 0.2 --n-jobs 10


.. raw:: latex

    \newpage

Use detect.py with trained model
--------------------------------

.. code-block:: console

    $ python3 trajectorynet/detect.py --help
        usage: detect.py [-h] --video VIDEO --outdir OUTDIR [--database] [--joblib] --yolo-model YOLO_MODEL [--iou IOU] [--score SCORE] [--device DEVICE] [--half] [--show] [--max-age MAX_AGE] [--model MODEL]

        Detect objects in a video.

        optional arguments:
        -h, --help            show this help message and exit
        --video VIDEO         Path to video file.
        --outdir OUTDIR       Path to output video file.
        --database            Save results to database.
        --joblib              Save results to database.
        --yolo-model YOLO_MODEL
                                Path to model weights file.
        --iou IOU             IoU threshold.
        --score SCORE         Score threshold.
        --device DEVICE       Device to run inference on.
        --half                Use half precision.
        --show                View output video.
        --max-age MAX_AGE     Max age of a track.
        --model MODEL         Path to trajectorynet model.

    python3 trajectorynet/detect.py --video ../../cv_research_video_dataset/Bellevue_116th_NE12th/Bellevue_116th_NE12th__2017-09-11_12-08-33.mp4 --yolo-model trajectorynet/yolov7/yolov7.pt --device 0 --show --half --max-age 30 --model ../../cv_research_video_dataset/Bellevue_116th_NE12th_24h/Preprocessed_threshold_0.7_enter-exit-distance_0.1/models/SVM_7.joblib