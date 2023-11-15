from argparse import ArgumentParser

from DetectionPipeline import DeepSORT, Detector, Yolov7, TrajectoryNet


def get_args():
    parser = ArgumentParser(description='Detect objects in a video.')
    # region general arguments
    parser.add_argument('--video', type=str, required=True, help='Path to video file.')
    parser.add_argument('--outdir', type=str, help='Path to output video file.')
    parser.add_argument('--database', action='store_true', default=False, help='Save results to database.')
    parser.add_argument('--joblib', action='store_true', default=True, help='Save results to database.')
    # endregion
    # region YoloV7 arguments
    parser.add_argument('--yolo-model', type=str, required=True, help='Path to model weights file.')
    parser.add_argument('--iou', type=float, default=0.45, help='IoU threshold.')
    parser.add_argument('--score', type=float, default=0.5, help='Score threshold.')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on.')
    parser.add_argument('--half', action='store_true', help='Use half precision.')
    parser.add_argument('--show', action='store_true', help='View output video.')
    # endregion
    # region deepsort arguments
    parser.add_argument('--max-age', type=int, default=30, help='Max age of a track.')
    # endregion
    # region trajectorynet arguments
    parser.add_argument('--model', type=str, help='Path to trajectorynet model.')
    # endregion
    return parser.parse_args()

def main():
    args = get_args()
    yolo = Yolov7(weights=args.yolo_model, conf_thres=args.score, iou_thres=args.iou, half=args.half, device=args.device, debug=True)
    deepSort = DeepSORT(args.max_age)
    detector = Detector(source=args.video, outdir=args.outdir, database=args.database, joblib=args.joblib, model=args.model)
    predictor = TrajectoryNet(model=args.model, debug=True)
    detector.run(yolo, deepSort, predictor, show=args.show)

    
if __name__ == '__main__':
    main()