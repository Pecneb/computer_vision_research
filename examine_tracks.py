from dataManagementClasses import TrackedObject, Detection
from processing_utils import load_joblib_tracks
from argparse import ArgumentParser
import cv2
import numpy as np

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
    cv2.putText(ret_img, "{} [{:.2f}] VX: {:.2f} VY: {:.2f} AX: {:.2f} AY: {:.2f}".format(detection.label, float(detection.confidence), float(detection.VX * image.shape[1] / aspect_ratio), float(detection.VY * image.shape[0]), float(detection.AX* image.shape[1] / aspect_ratio), float(detection.AY * image.shape[0])),
                (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,255,0), 2)
    return ret_img

def database_is_joblib(path: str):
    return path.split('.')[-1] == "joblib"

def draw_trajectory(img, track: TrackedObject, upscale: bool = True):
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

def examine_tracks(args):
    if not database_is_joblib(args.database):
        raise IOError(("Not joblib extension."))
    tracks = load_joblib_tracks(args.database)
    
    i_track = 0
    while i_track >= 0 and i_track < len(tracks):
        video = cv2.VideoCapture(args.video)
        if not video.isOpened():
            raise IOError("Can not open video.")

        start_frame = tracks[i_track].history[0].frameID
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame-1)
        act_start_frame = video.get(cv2.CAP_PROP_POS_FRAMES)+1

        if act_start_frame == start_frame:
            window_name = f"Object ID{tracks[i_track].objID}"
            cv2.namedWindow(window_name)
            i_det = 0
            i_frame = 0
            while i_frame < tracks[i_track].history[-1].frameID-tracks[i_track].history[0].frameID and i_frame >= 0:
                ret, frame = video.read()
                if not ret:
                    break

                act_frame_num = video.get(cv2.CAP_PROP_POS_FRAMES)
                frame_traj = draw_trajectory(frame, tracks[i_track])
                if act_frame_num == tracks[i_track].history[i_det].frameID:
                    frame_traj = drawbbox(
                        tracks[i_track].history[i_det],
                        frame_traj
                    )
                    i_det += 1

                cv2.imshow(window_name, frame_traj)

                i_frame += 1

                key_2 =cv2.waitKey(0)
                if  key_2 == ord('s'):
                    continue
                """if key_2 == ord('r'):
                    if i_frame - 1 >= 0:
                        video.set(cv2.CAP_PROP_POS_FRAMES, act_frame_num-2)
                        i_frame -= 1
                    continue"""
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
        prog="Program to examine trajectories individually.",
    )
    argparser.add_argument(
        "-db", "--database",
        type=str,
        required=True,
        help="Path to joblib database file."
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