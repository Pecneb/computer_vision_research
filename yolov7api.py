import argparse
import cv2 as cv
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import letterbox
from yolov7.utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh
from yolov7.utils.torch_utils import select_device, load_classifier

CONFIG = "yolov7/cfg/deploy/yolov7.yaml"
WEIGHTS = "yolov7/yolov7.pt"
IMGSZ = 640 
STRIDE = 32
DEVICE = "cuda" 
CLASSIFY = False 
AUGMENT = True
CONF_THRES = 0.35
IOU_THRES = 0.50
CLASSES = None

from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()


def load_model(device=DEVICE, weights=WEIGHTS, imgsz=IMGSZ, classify=CLASSIFY):
    free_gpu_cache()

    device = select_device(device, batch_size=1)
    half = device.type != 'cpu'

    # Start initializing our yolov7 model
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())
    imgsz = check_img_size(imgsz, s=stride)

    if half:
        model.half() # to FP16

    cudnn.benchmark = True # set True if img size is constant
    if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    # optional Second-stage classifier
    if classify:
        modelc = load_classifier(name='resnet101', n=2)
        modelc.load_state_dict(torch.load('yolov7/weights/resnet101.nt', map_location=device)['model']).to(device)
        return model, modelc, half, device 
    
    return model, half, device

if CLASSIFY:
    MODEL, MODELC, HALF, DEVICE = load_model()
else:
    MODEL, HALF, DEVICE = load_model()

NAMES = MODEL.module.names if hasattr(MODEL, "module") else MODEL.names
COLORS = {}
for name in NAMES:
    COLORS[name] = [random.randint(0, 255) for _ in range(3)]

def detect(img0, model=MODEL, modelc=None, half=HALF, imgsz=IMGSZ, stride=STRIDE, device=DEVICE, augment=AUGMENT, names=NAMES, colors=COLORS, conf_thres=CONF_THRES, iou_thres=IOU_THRESH, classes=CLASSES, classify=CLASSIFY):    
    with torch.no_grad():
        # Scale img0 to model imgsz
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :,::-1].transpose(2, 0, 1)
        img = torch.from_numpy(img.copy()).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension:
            img = img.unsqueeze(0)
        
        pred = model(img, augment=augment)[0] # we input only one image so we only need the first row of the detection matrix 
        pred = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres, classes=CLASSES) # remove low confidence detections from matrix
        
        if classify:
            pred = apply_classifier(pred, modelc, img, img0) # if classify is True, secondary classification on detection matrix 

        detections_adjusted = [] 
        for det in pred: 
            # Rescale boxes from img_size to img0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                bbox = xyxy2xywh(torch.tensor(xyxy).view(1, 4)) # convert to center coords, width, height format
                label = names[int(cls)]
                detections_adjusted.append([label, conf.item(), bbox[0,:].numpy()])
            
        
        return detections_adjusted 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--input', help='video source')
    args = parser.parse_args()
    with torch.no_grad():
        cap = cv.VideoCapture(args.input)
        if not cap.isOpened():
            print("Source cannot be opened.")
            exit(0)
        while(1):
            ret, frame = cap.read()
            if frame is None:
                break

            print(detect(frame, conf_thres=0.50))

            if cv.waitKey(1) == ord('q'):
                break

    
    