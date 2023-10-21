### Third Party ###
import os
from argparse import ArgumentParser

import cv2
import numpy as np
from classification import make_feature_vectors_version_one
from clustering import calc_cluster_centers, upscale_cluster_centers
from computer_vision_research.utility.dataManagementClasses import Detection, TrackedObject
from icecream import ic
### Local ###
from utility.dataset import load_dataset
from utility.models import load_model
from visualizer import draw_clusters, draw_prediction, draw_prediction_line

DEBUG = True if os.getenv("DEBUG") else False
if not DEBUG:
    ic.disable()

def upscalebbox(bbox, fwidth, fheight):
    """Upscale normalized coordinates to the video's frame size.
    The downscaling method was: X = X * fwidth / (fwidth/fheight), 
                                Y = Y * fheight
                                W = W * fwidth / (fwidth/fheight), 
                                H = H * fheight

    Args:
        bbox (tuple): Tuple of 4 values (X,Y,W,H) 
        fwidth (int): Frame width of the video. 
        fheight (_type_): Frame height of the video. 
    """
    ratio = fwidth / fheight
    X, Y, W, H = bbox
    X = (X * fwidth) / ratio 
    W = (W * fwidth) / ratio
    Y = Y * fheight 
    H = H * fheight 
    return X, Y, W, H

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def drawbbox(detection: Detection, image: np.ndarray):
    """Draw bounding box of an object to the given image.

    Args:
        detection (Detection): Detection object. 
        image (np.ndarray): OpenCV image object.
    """
    aspect_ratio = image.shape[1] / image.shape[0]
    bbox = (
        detection.X,
        detection.Y,
        detection.Width,
        detection.Height
        )
    ret_img = image.copy()
    bboxUpscaled = upscalebbox(bbox, image.shape[1], image.shape[0])
    left, top, right, bottom = bbox2points(bboxUpscaled)
    cv2.rectangle(ret_img, (left, top), (right, bottom), (0,255,0), 1)
    cv2.putText(ret_img, "{}".format(detection.label,), #float(detection.confidence), float(detection.VX * image.shape[1] / aspect_ratio), float(detection.VY * image.shape[0]), float(detection.AX* image.shape[1] / aspect_ratio), float(detection.AY * image.shape[0])),
                (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,0), 2)
    return ret_img

def database_is_joblib(path: str):
    return path.split('.')[-1] == "joblib"

def draw_trajectory(img, track: TrackedObject, upscale: bool = True, actual_detection_id: int = None):
    linesize = 3
    X = track.history_X
    Y = track.history_Y
    if upscale:
        aspect_ratio = img.shape[1] / img.shape[0]
        X = X * (img.shape[1] / aspect_ratio)
        Y = Y * img.shape[0]
    ret_img = img.copy()
    pts = np.column_stack((X, Y)).astype(int) 
    cv2.polylines(ret_img, [pts], False, (0,255,0))
    for i in range(X.shape[0]):
        if i == 0:
            cv2.circle(ret_img, (int(X[i]), int(Y[i])), linesize, (0,255,0), -1)
        elif i == X.shape[0]-1:
            cv2.circle(ret_img, (int(X[i]), int(Y[i])), linesize, (0,0,255), -1)
        elif i == X.shape[0]//2:
            cv2.circle(ret_img, (int(X[i]), int(Y[i])), linesize, (0,255,255), -1)
        else:
            cv2.circle(ret_img, (int(X[i]), int(Y[i])), linesize, (255,0,0), -1)
    if actual_detection_id is not None:
        cv2.circle(ret_img, (int(X[actual_detection_id]), int(Y[actual_detection_id])), linesize, (255,255,0), -1)
    return ret_img

def next_frame_id(i_frame: int, max_i: int):
    """If actial frame id is less or equal to the max id,
    then return next frame id, -1 otherwise.

    Args:
        i_frame (int): Actual frame id. 
        max_i (int): Maximum limit frame id. 

    Returns:
        int: Next frame id or -1, if there is no next frame. 
    """
    ret_i = i_frame + 1
    if ret_i > max_i:
        return -1
    return ret_i

def previous_frame_id(i_frame: int, min_i: int):
    """If actial frame id is larger than minimum id,
    then return previous frame id, -1 otherwise.

    Args:
        i_frame (int): Actual frame id. 
        min_i (int): Minimum limit frame id. 

    Returns:
        int: Previous frame id or -1, if there is no previous frame. 
    """
    ret_i = i_frame - 1
    if ret_i < min_i:
        return -1
    return ret_i

def print_object_info(obj: TrackedObject, i_det: int, img: np.ndarray):
    aspect_ratio = img.shape[1] / img.shape[0]
    print("\nDetection number: {:3} X: {:10.6f}, Y: {:10.6f}, VX: {:10.6f}, VY: {:10.6f}, AX: {:10.6f}, AY: {:10.6f}".format(
        i_det,
        obj.history_X[i_det] * img.shape[1] / aspect_ratio, 
        obj.history_Y[i_det] * img.shape[0], 
        obj.history_VX_calculated[i_det] * img.shape[1] / aspect_ratio, 
        obj.history_VY_calculated[i_det] * img.shape[0], 
        obj.history_AX_calculated[i_det] * img.shape[1] / aspect_ratio, 
        obj.history_AY_calculated[i_det] * img.shape[0]))

def make_feature_vectors(track: TrackedObject, max_history_len: int = 30) -> np.ndarray:
    feature_vectors = []
    frame_number = []
    for i in range(2, len(track.history)-1):
        first_idx = 0 if i < max_history_len else i - max_history_len
        middle_idx = i // 2
        ic(first_idx)
        ic(middle_idx)
        ic(i)
        feature_vectors.append(np.array([track.history_X[first_idx], track.history_Y[first_idx],
                                track.history_VX_calculated[first_idx], track.history_VY_calculated[first_idx],
                                track.history_X[middle_idx], track.history_Y[middle_idx],
                                track.history_X[i], track.history_Y[i],
                                track.history_VX_calculated[i], track.history_VY_calculated[i]]))
        frame_number.append(track.history[i].frameID)
    return np.array(feature_vectors), np.array(frame_number)

def examine_tracks(args):
    if not database_is_joblib(args.database):
        raise IOError(("Not joblib extension."))
    tracks = load_dataset(args.database)
    model = load_model(args.model)
    i_track = 0
    while i_track >= 0 and i_track < len(tracks):
        video = cv2.VideoCapture(args.video)
        if not video.isOpened():
            raise IOError("Can not open video.")

        start_frame = tracks[i_track].history[0].frameID
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
        act_start_frame = video.get(cv2.CAP_PROP_POS_FRAMES)+1
        frame_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        centroids_upscaled = upscale_cluster_centers(model.centroid_coordinates, frame_width, frame_height)

        if act_start_frame == start_frame:
            window_name = f"Object ID{tracks[i_track].objID}"
            cv2.namedWindow(window_name)
            i_det = 0
            i_frame = 0
            act_frame_num = video.get(cv2.CAP_PROP_FRAME_COUNT)
            # feature_vs, _, metadata, _ = make_feature_vectors_version_one([tracks[i_track]], k=6)
            feature_vs, frame_numbers = make_feature_vectors(tracks[i_track])
            while i_frame < tracks[i_track].history[-1].frameID-tracks[i_track].history[0].frameID and i_frame >= 0:
                ret, frame = video.read()
                if not ret:
                    break

                # Make frame transparent
                frame = 255 - (255 - frame) * 0.5
                frame = frame.astype(np.uint8)

                draw_clusters(centroids_upscaled, frame)

                act_frame_num = video.get(cv2.CAP_PROP_POS_FRAMES)

                fv = None
                for i in range(len(feature_vs)):
                    if act_frame_num == frame_numbers[i]:
                        fv = feature_vs[i]

                if act_frame_num == tracks[i_track].history[i_det].frameID:
                    print_object_info(tracks[i_track], i_det, frame)
                    frame_traj = draw_trajectory(frame, tracks[i_track], actual_detection_id=i_det)
                    frame_traj = drawbbox(
                        tracks[i_track].history[i_det],
                        frame_traj
                    )
                    if fv is not None:
                        #TODO actual feature vector, not last
                        pred_proba = model.predict_proba(fv.reshape(1, -1), classes=model.centroid_labels)
                        pred = np.argsort(pred_proba)[:, -1:] # -args.top_k:]
                        pred = pred.reshape((-1))
                        ic(pred)
                        ic(pred_proba)
                        fv_upscaled = tracks[i_track].upscale_feature(fv, frame_width, frame_height)
                        draw_prediction_line(frame_traj, centroids_upscaled, pred[0], (fv_upscaled[-4], fv_upscaled[-3]))

                    i_det += 1
                else:
                    frame_traj = draw_trajectory(frame, tracks[i_track])

                cv2.imshow(window_name, frame_traj)

                i_frame += 1

                key_2 =cv2.waitKey(0)
                if  key_2 == ord('s'):
                    continue
                if key_2 == ord('r'):
                    if i_det - 1 >= 0:
                        i_det -= 2 
                        video.set(cv2.CAP_PROP_POS_FRAMES, tracks[i_track].history[i_det].frameID-1)
                        i_frame = tracks[i_track].history[i_det].frameID - tracks[i_track].history[0].frameID
                    continue
                if key_2 == ord('b') or key_2 == ord('q') or key_2 == ord('n'):
                    break
        else:
            i_track += 1
            cv2.destroyAllWindows()
            continue

        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
        if key == ord('r'):
            cv2.destroyAllWindows()
            continue
        if key == ord('n'):
            if i_track + 1 < len(tracks):
                i_track += 1
            else:
                print("Cant access next track.")
            cv2.destroyAllWindows()
            continue
        if key == ord('b'):
            if i_track - 1 >= 0:
                i_track -= 1
            else:
                print("Cant access previous track.")
            cv2.destroyAllWindows()
            continue
            

def main():
    argparser = ArgumentParser(
        prog="Program to examine trajectories individually.\n"
             "\tStep between objects with 'n' to move forward 'b' to move backward.\n"
             "\tStep between frames and detections with 's' to step forward 'r' to"
             "step backward.\n"
             "\tPress 'q' to exit.\n"
    )
    argparser.add_argument(
        "-db", "--database",
        type=str,
        required=True,
        help="Path to joblib database file."
    )
    argparser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path to model."
    )
    argparser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video srouce."
    )
    argparser.set_defaults(func=examine_tracks)

    args = argparser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()