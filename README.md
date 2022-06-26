# computer_vision_research

Predicting trajectories of objects

**TODO:** implement dataclasses to store objects and detections  
**TODO:** implement SQLite DB logging  
**TODO:** implement linear regression to predict future position of objects  

## Darknet

For detection, I used darknet neural net and YOLOV4 pretrained model. [[1]](#1)
In order to be able to use the darknet api, build from source with the LIB flag on. Then copy libdarknet.so to root dir of the project.

## References
<a id="1">[1]</a>  
@misc{bochkovskiy2020yolov4,  
      title={YOLOv4: Optimal Speed and Accuracy of Object Detection},  
      author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},  
      year={2020},  
      eprint={2004.10934},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV}  
}  
@InProceedings{Wang_2021_CVPR,  
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},  
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},  
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},  
    month     = {June},  
    year      = {2021},  
    pages     = {13029-13038}  
}
