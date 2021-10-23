from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
from pandas import DataFrame
from cv2 import cv2
from imutils import grab_contours
import numpy as np
from typing import Tuple, Optional, List
from utils import crop_image, clean_digit_cell


def convert_image_to_sudoku(image_path: str, debug: bool = False) -> np.array:
    img = cv2.imread(filename=image_path)
    if debug:
        cv2.imshow('original', img)
        cv2.waitKey(delay=0)
    img, img_reverted = crop_image(image=img, debug=debug)
    if debug:
        cv2.imshow('cropped', img)
        cv2.waitKey(delay=0)
    create_array_of_digit_images(image=img, debug=True)
    return DataFrame()


def create_array_of_digit_images(image: np.ndarray, sudoku_size: int = 9, debug: bool = False) -> List[List[np.ndarray]]:
    h, _, _ = image.shape
    step_size = int(h / sudoku_size)

    cell_list = []
    for i in range(sudoku_size):
        cell_row = []
        for j in range(sudoku_size):
            start_y = i * step_size
            start_x = j * step_size
            end_x = start_x + step_size
            end_y = start_y + step_size
            cell_image = image[start_y:end_y, start_x:end_x]
            if debug:
                cv2.imshow(f'cell ({start_x}, {start_y})', cell_image)
                cv2.waitKey(delay=0)
            cell_row.append(cell_image)
        cell_list.append(cell_row)
    return cell_list


convert_image_to_sudoku(r"E:\Programming\Repos\sudoku_app\server\dataset\image1.jpg")
