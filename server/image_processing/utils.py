from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from cv2 import cv2
from imutils import grab_contours
import numpy as np
from typing import Tuple, Optional


def crop_image(image: np.ndarray, debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    max_area = image.shape[0] * image.shape[1] / 4
    if debug:
        print(f'image shape is: {image.shape[:2]}, max area of the image is: {max_area}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 57, 5)
    cv2.imshow('board_thresh', thresh)
    cv2.waitKey(delay=0)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(cnts=contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    board_cnt = None
    for c in contours:
        length = cv2.arcLength(curve=c, closed=True)
        approx = cv2.approxPolyDP(curve=c, epsilon=0.02 * length, closed=True)

        # Found a four point contour, most likely the board itself because the list of cnts is sorted
        if len(approx) == 4:
            board_cnt = approx
            break
    if board_cnt is None:
        raise ValueError('Sudoku board was not found')

    if debug:
        image_copy = image.copy()
        cv2.drawContours(image_copy, [board_cnt], -1, (0, 255, 0))
        cv2.imshow('board', image_copy)
        cv2.waitKey(delay=0)

    only_board = four_point_transform(image, board_cnt.reshape(4, 2))
    only_board_inverted = four_point_transform(thresh, board_cnt.reshape(4, 2))

    return only_board, only_board_inverted


def clean_digit_cell(cell: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    thresh = clear_border(thresh)

    if debug:
        cv2.imshow('cell_thresh', thresh)
        cv2.waitKey(delay=0)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # Here in case there is no digit
        return None

    c = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(image=mask, contours=[c], contourIdx=-1, color=255, thickness=-1)

    h, w = thresh.shape
    percent_filled = cv2.countNonZero(src=mask) / float(w * h)

    if percent_filled < 0.03:
        # Again, probably an empty cell in this case
        return None

    digit_cell = cv2.bitwise_and(src1=thresh, src2=thresh, mask=mask)

    if debug:
        cv2.imshow('cell_cleared', digit_cell)
        cv2.waitKey(delay=0)

    return digit_cell
