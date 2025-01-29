# utils for project

import sys
from os import system
from pathlib import Path
from shutil import copy

import cv2 as cv
import numpy as np
from PIL import Image

path_to_eyes = "/dcs/large/u2204489/eyes"


def check_300w_dataset() -> None:
    """the 300w dataset can have either 68 or 51 landmarks, check all 68"""
    files = list(Path(f"{path_to_eyes}/300w/").glob("*.pts"))
    for file in files:
        with file.open() as f:
            lines = f.read().split("\n")
            n_points = int(lines[1].split(":")[1])
            if n_points != 68:  # noqa: PLR2004
                print(f"{file} has {n_points} points")
    print("done")


def visualise_eye_datasets() -> None:
    """visualise the eye datasets on sample images"""
    # 300w
    # visualise_points(f"{path_to_eyes}/300w/indoor_001", "png")
    visualise_bounding_boxes(f"{path_to_eyes}/300w/indoor_001", "png")
    visualise_bounding_boxes(f"{path_to_eyes}/300w/indoor_001_r", "png")
    # aflw
    # visualise_points(f"{path_to_eyes}/aflw/image00002", "jpg")
    visualise_bounding_boxes(f"{path_to_eyes}/aflw/image00002_01", "jpg")
    visualise_bounding_boxes(f"{path_to_eyes}/aflw/image00002_01_r", "jpg")
    # afw
    # visualise_points(f"{path_to_eyes}/afw/134212_1", "jpg")
    visualise_bounding_boxes(f"{path_to_eyes}/afw/134212_1", "jpg")
    visualise_bounding_boxes(f"{path_to_eyes}/afw/134212_1_r", "jpg")
    # helen
    # visualise_points(f"{path_to_eyes}/helen/12799337_1", "jpg")
    visualise_bounding_boxes(f"{path_to_eyes}/helen/12799337_1", "jpg")
    visualise_bounding_boxes(f"{path_to_eyes}/helen/12799337_1_r", "jpg")
    # lfpw
    # visualise_points(f"{path_to_eyes}/lfpw/trainset/image_0001", "png")
    visualise_bounding_boxes(f"{path_to_eyes}/lfpw/trainset/image_0001", "png")
    visualise_bounding_boxes(f"{path_to_eyes}/lfpw/trainset/image_0001_r", "png")


def visualise_points(
    path: str, image_type: str, points: np.ndarray | None = None
) -> None:
    """draw labelled points on an image"""
    # if no points provided, use the pts file
    if points is None:
        with Path(f"{path}.pts").open() as f:
            points = np.loadtxt(f, comments=("version:", "n_points:", "{", "}"))
    img = cv.imread(f"{path}.{image_type}")
    # draw labelled points
    for i, (x, y) in enumerate(points):
        cv.circle(img, (int(float(x)), int(float(y))), 1, (0, 0, 255), -1)
        cv.putText(
            img,
            str(i),
            (int(float(x)), int(float(y))),
            cv.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )
    cv.imshow("image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def visualise_bounding_boxes(path: str, image_type: str) -> None:
    """draw bounding boxes and eye points on an image"""
    with Path(f"{path}.txt").open() as f:
        lines = f.read().split("\n")
        img = cv.imread(f"{path}.{image_type}")
        # draw bounding boxes
        for i in range(6):
            x, y, w, h = cxcywh_to_xywh(list(map(float, lines[i].split())))
            cv.rectangle(
                img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 1
            )
        # draw eye points (if applicable)
        if len(lines) == 18:  # noqa: PLR2004
            for i in range(6, 18):
                x, y = map(float, lines[i].split())
                cv.circle(img, (int(x), int(y)), 1, (0, 0, 255), -1)
        cv.imshow("image", img)
        cv.waitKey(0)
        cv.destroyAllWindows()


# points are of form [left eye, right eye, left eyebrow, right eyebrow, nose, mouth]
# multi_pie
# [[36:41],
#  [45, 44, 43, 42, 47, 46],
#  [17:21],
#  [22:26],
#  [27:35],
#  [48:59]]

# custom txt is bounding boxes in x, y, w, h format
# followed by x, y points of eye landmarks


def format_eye_datasets() -> None:
    """converts from pts to custom txt files"""
    multi_pie_landmarks = [
        list(range(36, 42)),
        [45, 44, 43, 42, 47, 46],
        list(range(17, 22)),
        list(range(22, 27)),
        list(range(27, 36)),
        list(range(48, 60)),
    ]
    format_eye_dataset(f"{path_to_eyes}/300w/", multi_pie_landmarks)
    format_eye_dataset(f"{path_to_eyes}/afw/", multi_pie_landmarks)
    format_eye_dataset(f"{path_to_eyes}/helen/", multi_pie_landmarks)
    format_eye_dataset(f"{path_to_eyes}/lfpw/testset/", multi_pie_landmarks)
    format_eye_dataset(f"{path_to_eyes}/lfpw/trainset/", multi_pie_landmarks)
    reformat_aflw_dataset()


def format_eye_dataset(
    path: str,
    landmark_indices: list[list[int]],
) -> None:
    """converts from pts to custom txt files"""
    files = list(Path(path).glob("*.pts"))
    for file in files:
        print("Processing file:", file)
        points = np.loadtxt(file, comments=("version:", "n_points:", "{", "}"))

        file_content = ""
        # create bounding box
        for landmark in landmark_indices:
            landmarks = points[landmark]
            x, y, w, h = get_rectangle(landmarks)
            file_content += f"{x} {y} {w} {h}\n"
        # combine eye landmarks
        eye_landmarks = np.append(
            points[landmark_indices[0]], points[landmark_indices[1]], axis=0
        )
        for x, y in eye_landmarks:
            file_content += f"{x} {y}\n"
        # save to file
        with file.with_suffix(".txt").open("w") as f:
            f.write(file_content.strip())


def get_rectangle(points: np.ndarray) -> tuple[float, float, float, float]:
    """find bounding rectangle of points"""
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    centre_x = (x_min + x_max) / 2
    centre_y = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min
    return centre_x, centre_y, width, height


def reformat_aflw_dataset() -> None:
    """converts aflw into bounding boxes"""
    # convert pngs to jpgs
    imgs = list(Path(f"{path_to_eyes}/aflw/").glob("*.png"))
    for img in imgs:
        print("Converting:", img)
        jpg = Image.open(img).convert("RGB")
        jpg.save(f"{img.with_suffix('.jpg')}")
        img.unlink()

    # delete all images without a txt file
    imgs = list(Path(f"{path_to_eyes}/aflw/").glob("*.jpg"))
    for img in imgs:
        # txts are of form "image00002_01.txt"
        if not Path(f"{path_to_eyes}/aflw/{img.stem}_01.txt").exists():
            print("Deleting:", img)
            img.unlink()

    aflw_landmarks = [
        [7, 8, 9, 46, 47, 48, 49, 82, 83],
        [10, 11, 12, 50, 51, 52, 53, 84, 85],
        [1, 2, 3, 36, 37, 72, 73, 74],
        [4, 5, 6, 38, 39, 75, 76, 77],
        [14, 15, 16, 40, 41, 42, 43, 44, 45, 78, 79, 80, 81],
        [18, 20, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63],
    ]
    aflw_landmarks = [[i - 1 for i in landmark] for landmark in aflw_landmarks]

    files = list(Path(f"{path_to_eyes}/aflw/").glob("*.txt"))
    for file in files:
        print("Processing file:", file)
        points = np.loadtxt(file, skiprows=1)
        file_content = ""
        for landmark in aflw_landmarks:
            landmarks = points[landmark]
            x, y, w, h = get_rectangle(landmarks)
            file_content += f"{x} {y} {w} {h}\n"
        with Path(file).open("w") as f:
            f.write(file_content.strip())

    # each image (for example image00002.jpg) can have >= 1 txt files related to it
    # rename the images, to match, copying the image if necessary
    files = list(Path(f"{path_to_eyes}/aflw/").glob("*.txt"))
    for file in files:
        print("Convert file:", file)
        image = Path(f"{path_to_eyes}/aflw/{file.stem.split('_')[0]}.jpg")
        copy(image, f"{file.with_suffix('.jpg')}")

    # delete all images without a "_"
    imgs = list(Path(f"{path_to_eyes}/aflw/").glob("*.jpg"))
    for img in imgs:
        if "_" not in img.stem:
            print("Deleting:", img)
            img.unlink()


def reflect_datasets() -> None:
    """reflects the datasets horizontally to double the size"""
    # delete all files with "*_r_r*"
    # oops
    system("find /dcs/large/u2204489/eyes -type f -name '*_r_r*' -exec rm -f {} +")

    reflect_dataset(f"{path_to_eyes}/300w/", "png")
    reflect_dataset(f"{path_to_eyes}/aflw/", "jpg")
    reflect_dataset(f"{path_to_eyes}/afw/", "jpg")
    reflect_dataset(f"{path_to_eyes}/helen/", "jpg")
    reflect_dataset(f"{path_to_eyes}/lfpw/testset/", "png")
    reflect_dataset(f"{path_to_eyes}/lfpw/trainset/", "png")


# this takes a while, could be sped up with parallelism
# but its only run once so not a big deal
# also NFS might complain if hit it too hard
def reflect_dataset(path: str, type: str) -> None:
    """reflects an individual dataset horizontally"""

    files = list(Path(path).glob("*.txt"))
    for file in files:
        print("Reflecting file:", file)
        # reflect the image
        # (thanks to python3.12 for embedded f-strings)
        img = cv.imread(f"{file.with_suffix(f".{type}")}")
        reflected_img = cv.flip(img, 1)
        cv.imwrite(str(file.with_name(file.stem + f"_r.{type}")), reflected_img)

        with Path(file).open() as f:
            lines = f.read().split("\n")

        file_string = ""
        # reflect bounding boxes
        for i in range(6):
            x, y, w, h = map(float, lines[i].split())
            file_string += f"{img.shape[1] - x} {y} {w} {h}\n"
        # reflect eye points (if applicable)
        if len(lines) == 18:  # noqa: PLR2004
            for i in range(6, 18):
                x, y = map(float, lines[i].split())
                file_string += f"{img.shape[1] - x} {y}\n"

        with file.with_name(file.stem + "_r.txt").open("w") as f:
            f.write(file_string.strip())


def convert_dataset_coords() -> None:
    convert_coords(f"{path_to_eyes}/300w/")
    convert_coords(f"{path_to_eyes}/aflw/")
    convert_coords(f"{path_to_eyes}/afw/")
    convert_coords(f"{path_to_eyes}/helen/")
    convert_coords(f"{path_to_eyes}/lfpw/testset/")
    convert_coords(f"{path_to_eyes}/lfpw/trainset/")
    print("done :)")


def convert_coords(path: str) -> None:
    """converts bounding box of x,y,w,h to center_x, center_y, w, h"""

    files = list(Path(path).glob("*.txt"))
    for file in files:
        with file.open() as f:
            lines = f.read().split("\n")
        # bounding boxes are the first 6 lines
        bounding_boxes = [list(map(float, line.split())) for line in lines[:6]]
        # eye points are the next 12 lines (if applicable)
        eye_points = [list(map(float, line.split())) for line in lines[6:]]
        file_string = ""
        for x, y, w, h in bounding_boxes:
            file_string += f"{x + w / 2} {y + h / 2} {w} {h}\n"
        for x, y in eye_points:
            file_string += f"{x} {y}\n"
        with file.open("w") as f:
            f.write(file_string.strip())


def cxcywh_to_xywh(initial_box: list[float]) -> list[float]:
    """converts bounding box of center_x, center_y, w, h to x,y,w,h"""
    cx, cy, w, h = initial_box
    return [cx - w / 2, cy - h / 2, w, h]


if __name__ == "__main__":
    arg = None
    try:
        arg = sys.argv[1]
    except IndexError:
        print("invalid argument")
        sys.exit(1)

    if arg == "aflw":
        reformat_aflw_dataset()
    elif arg == "visualise":
        visualise_eye_datasets()
    elif arg == "300w":
        check_300w_dataset()
    elif arg == "format":
        format_eye_datasets()
    elif arg == "reflect":
        reflect_datasets()
    elif arg == "convert":
        convert_dataset_coords()
    else:
        print("invalid argument")
        sys.exit(1)
