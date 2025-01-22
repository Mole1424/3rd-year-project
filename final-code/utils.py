# utils for project

import sys
from os import system
from pathlib import Path

import cv2 as cv
import numpy as np
from scipy.io import loadmat

path_to_eyes = "/dcs/large/u2204489/eyes"


def format_helen_dataset() -> None:
    """reformats the helen dataset annotations to pts files"""
    num_files = 2330

    # landmarks are lines 1-194
    landmarks = list(range(1, 195))

    # iterate over all txt files and save the landmarks to pts files
    for i in range(1, num_files + 1):
        with Path(f"{path_to_eyes}/helen/{i}.txt").open() as f:
            lines = f.read().split("\n")
            image_title = lines[0]

            # reformat pts if image exists
            if Path(f"{path_to_eyes}/helen/{image_title}.jpg").exists():
                # save landmarks to pts file
                with Path(f"{path_to_eyes}/helen/{image_title}.pts").open("w") as w:
                    w.write("version: 1\n")
                    w.write(f"n_points: {len(landmarks)}\n")
                    w.write("{\n")
                    for landmark in landmarks:
                        x, y = lines[landmark].split(" , ")
                        w.write(f"{x} {y}\n")
                    w.write("}")
        # remove old txt file
        Path(f"{path_to_eyes}/helen/{i}.txt").unlink()


def format_aflw_dataset() -> None:
    """reformats the aflw dataset annotations to pts files"""
    # load mat annotations
    annotations = loadmat(f"{path_to_eyes}/aflw/AFLWinfo_release.mat")

    # loop over all annotated images
    for i in range(len(annotations["nameList"])):
        image_title = annotations["nameList"][i][0][0].split("/")[-1].split(".")[0]

        if Path(f"{path_to_eyes}/aflw/{image_title}.jpg").exists():
            # convert landmarks from [x1, x2, ..., y1, y2, ...] to [x1, y1, ...]
            landmarks = annotations["data"][i]
            points = list(zip(landmarks[:19], landmarks[19:]))
            # save landmarks to pts file
            with Path(f"{path_to_eyes}/aflw/{image_title}.pts").open("w") as f:
                f.write("version: 1\n")
                f.write(f"n_points: {len(points)}\n")
                f.write("{\n")
                for x, y in points:
                    f.write(f"{x} {y}\n")
                f.write("}")


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
    visualise_bounding_boxes(f"{path_to_eyes}/aflw/image00002", "jpg")
    visualise_bounding_boxes(f"{path_to_eyes}/aflw/image00002_r", "jpg")
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
            x, y, w, h = map(float, lines[i].split())
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
# 300w, afw, and lfpw
# [[36:41],
#  [45, 44, 43, 42, 47, 46],
#  [17:21],
#  [22:26],
#  [27:35],
#  [48:59]]
# helen
# [[144, 141, 137, 134, 151, 147],
#  [125, 122, 117, 115, 131, 128],
#  [174:193]
#  [154:173]
#  [41:57]
#  [58:85, midpoint of 134, 114]

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
    helen_landmarks = [
        [144, 141, 137, 134, 151, 147],
        [125, 122, 117, 115, 131, 128],
        list(range(174, 194)),
        list(range(154, 174)),
        list(range(41, 58)),
        list(range(58, 86)),
    ]
    format_eye_dataset(f"{path_to_eyes}/300w/", multi_pie_landmarks)
    format_eye_dataset(f"{path_to_eyes}/afw/", multi_pie_landmarks)
    format_eye_dataset(f"{path_to_eyes}/helen/", helen_landmarks)
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
            if landmark[-1] == 85:  # noqa: PLR2004
                x1, y1 = points[134]
                x2, y2 = points[114]
                midpoint = [(x1 + x2) / 2, (y1 + y2) / 2]
                landmarks = np.vstack([landmarks, midpoint])
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
    return x_min, y_min, x_max - x_min, y_max - y_min


def reformat_aflw_dataset() -> None:
    """converts aflw into bounding boxes"""
    files = list(Path(f"{path_to_eyes}/aflw/").glob("*.pts"))
    landmarks = [
        [6, 7, 8],
        [9, 10, 11],
        [0, 1, 2],
        [3, 4, 5],
        [12, 13, 14],
        [15, 16, 17],
    ]
    for file in files:
        print("Processing file:", file)
        points = np.loadtxt(file, comments=("version:", "n_points:", "{", "}"))
        file_content = ""
        # create bounding box
        # the average eye width to height ratio is 0.353 (https://www.researchgate.net/figure/Average-eye-index-average-width-and-average-height-along-with-the-eye-classifications_tbl1_289499995)
        for landmark in landmarks[:2]:
            file_content += "{} {} {} {}\n".format(
                *create_weighted_box(points[landmark], 0.353)
            )
        # eyebrows can just take the bounding box of the points
        for landmark in landmarks[2:4]:
            file_content += "{} {} {} {}\n".format(*get_rectangle(points[landmark]))
        # nose
        nose_points = points[landmarks[4]]
        x1, y1 = points[8]
        x2, y2 = points[9]
        midpoint = [(x1 + x2) / 2, (y1 + y2) / 2]
        nose_points = np.vstack([nose_points, midpoint])
        file_content += "{} {} {} {}\n".format(*get_rectangle(nose_points))
        # mouth
        file_content += "{} {} {} {}\n".format(
            *create_weighted_box(points[landmarks[5]], 1)
        )
        # save to file
        with file.with_suffix(".txt").open("w") as f:
            f.write(file_content.strip())


def create_weighted_box(
    points: np.ndarray, ratio: float = 1
) -> tuple[float, float, float, float]:
    """create a bounding box with a given height to width ratio"""
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    width = x_max - x_min
    height = width * ratio
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    return x_mid - width / 2, y_mid - height / 2, width, height


def reflect_datasets() -> None:
    """reflects the datasets horizontally to double the size"""
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
    # delete all files with "*_r_r*"
    # oops
    system("find /dcs/large/u2204489/eyes -type f -name '*_r_r*' -exec rm -f {} +")

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
            file_string += f"{img.shape[1] - x - w} {y} {w} {h}\n"
        # reflect eye points (if applicable)
        if len(lines) == 18:  # noqa: PLR2004
            for i in range(6, 18):
                x, y = map(float, lines[i].split())
                file_string += f"{img.shape[1] - x} {y}\n"

        with file.with_name(file.stem + "_r.txt").open("w") as f:
            f.write(file_string.strip())


if __name__ == "__main__":
    arg = None
    try:
        arg = sys.argv[1]
    except IndexError:
        print("invalid argument")
        sys.exit(1)

    if arg == "helen":
        format_helen_dataset()
    elif arg == "aflw":
        format_aflw_dataset()
    elif arg == "visualise":
        visualise_eye_datasets()
    elif arg == "300w":
        check_300w_dataset()
    elif arg == "format":
        format_eye_datasets()
    elif arg == "reflect":
        reflect_datasets()
    else:
        print("invalid argument")
        sys.exit(1)
