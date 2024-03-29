import cv2
import matplotlib.pyplot as plt
import numpy as np


def masker(img: np.ndarray):
    img_cp = img.copy()
    drawing = np.zeros_like(img)
    # create mask of ones, that indicates a white board
    mask = np.ones(shape=img.shape[:2], dtype=np.uint8)
    # lists to store bounding box coordinates
    coordinates = []

    # function which will be called on mouse input
    def drawMaskCorners(action, x, y, flags, *userdata):
        # Mark top left corner when left mouse button is pressed
        if action == cv2.EVENT_LBUTTONUP:
            coordinates.append([x, y])

    # Create named window
    cv2.namedWindow("Window")
    # highgui function called when mouse events occur
    cv2.setMouseCallback("Window", drawMaskCorners)

    while (1):
        # Draw circles
        for c in coordinates:
            cv2.circle(drawing, (c[0], c[1]), 5, (0, 255, 0), -1)
        # Display image
        cv2.imshow("Window", cv2.add(img, drawing))
        # close the window when key q is pressed
        if cv2.waitKey(20) == ord('q'):
            break
        # If c is pressed clear the window, using the dummy image
        if cv2.waitKey(20) == ord('c'):
            drawing = np.zeros_like(img)
            coordinates = []
        # Remove last added circle
        if cv2.waitKey(20) == ord('b'):
            if len(coordinates) > 0:
                del coordinates[-1]
                drawing = np.zeros_like(img)
            else:
                print("No coordinates to remove.")

    remainder = len(coordinates) % 4
    if remainder != 0:
        coordinates = coordinates[:-remainder]

    # Create bbox corner cordinates from input point coordinates
    corners = np.array(coordinates).reshape(-1, 4, 2)
    for i in range(corners.shape[0]):
        # Calculate of the rect that our points can be fit in
        rect = cv2.minAreaRect(corners[i])
        # Create perpendicular rectangles
        box = cv2.boxPoints(rect)
        box = np.intp(box)  # convert to whole integers
        # cv2.drawContours(img, [box], 0, (0,255,0))
        cv2.fillPoly(mask, [box], (0, 0, 0))

    # apply mask to see masking results
    masked_img = cv2.bitwise_or(img_cp, img_cp, mask=mask)
    plt.imshow(cv2.cvtColor(masked_img, cv2.COLOR_BGR2RGB))
    plt.show()
    cv2.destroyAllWindows()
    return mask


def main():
    # Fake img
    # img = np.ones(shape=(512,512,3), dtype=np.uint8) * 255
    cap = cv2.VideoCapture(
        "/media/pecneb/DataStorage/computer_vision_research_test_videos/test_videos/stmarc_video.avi")
    ret, I = cap.read()
    mask = masker(I)
    plt.imshow(mask)
    plt.show()


if __name__ == "__main__":
    main()
