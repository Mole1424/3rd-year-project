# utils for project

import sys
from os import system
from pathlib import Path

import cv2 as cv
import numpy as np

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
    visualise_points(f"{path_to_eyes}/300w/indoor_001", "png")
    # afw
    visualise_points(f"{path_to_eyes}/afw/134212_1", "jpg")
    # cofw
    visualise_points(f"{path_to_eyes}/cofw/61", "jpg")
    # helen
    visualise_points(f"{path_to_eyes}/helen/12799337_1", "jpg")
    # ibug
    visualise_points(f"{path_to_eyes}/ibug/image_003_1", "jpg")
    # lfpw
    visualise_points(f"{path_to_eyes}/lfpw/trainset/image_0001", "png")
    # wflw
    visualise_points(f"{path_to_eyes}/wflw/0_Parade_marchingband_1_116", "jpg")


def visualise_points(path: str, image_type: str) -> None:
    """draw labelled points on an image"""
    with Path(f"{path}.pts").open() as f:
        points = np.loadtxt(f, comments=("version:", "n_points:", "{", "}"))
    img = cv.imread(f"{path}.{image_type}")
    # draw labelled points
    for i, (x, y) in enumerate(points):
        cv.circle(img, (int(float(x)), int(float(y))), 1, (0, 0, 255), -1)
        cv.putText(
            img,
            str(i + 1),
            (int(float(x)), int(float(y))),
            cv.FONT_HERSHEY_SIMPLEX,
            0.3,
            (255, 255, 255),
            1,
        )
    cv.imshow("image", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def format_cofw_dataset() -> None:
    """format cofw dataset to .pts format"""
    cofw_path = f"{path_to_eyes}/cofw/"

    main_pts_file = cofw_path + "COFW_test.txt"

    with Path(main_pts_file).open() as f:
        lines = f.read().split("\n")

    for i, line in enumerate(lines):
        print(f"Extracting file {i + 1}/{len(lines)}")
        line_arr = line.strip().split(" ")
        img = line_arr[0].split(".")[0]
        pts = line_arr[15:]
        with Path(cofw_path + img + ".pts").open("w") as f:
            f.write("version: 1\n")
            f.write(f"n_points: {int(len(pts) / 2)}\n")
            f.write("{\n")
            for j in range(0, len(pts), 2):
                f.write(f"{pts[j]} {pts[j + 1]}\n")
            f.write("}")

    print("done :)")


def format_wflw_dataset() -> None:
    """format wflw dataset to .pts format"""
    wflw_path = f"{path_to_eyes}/wflw/"

    landmarks_path = wflw_path + "landmarks.txt"

    # fmt: off
    # mapping from 98 points to 68 points
    points_map = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
                 33, 34, 35, 36, 37,
                 42, 43, 44, 45, 46,
                 51, 52, 53, 54, 55, 56, 57, 58, 59,
                 60, 61, 63, 64, 65, 67,
                 68, 69, 71, 72, 73, 75,
                 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                 88, 89, 90, 91, 92, 93, 94, 95]
    # fmt: on

    with Path(landmarks_path).open() as f:
        lines = f.read().split("\n")

    for i, line in enumerate(lines):
        print(f"Extracting file {i + 1}/{len(lines)}")
        line_arr = line.strip().split(" ")
        img = line_arr[-1].split("/")[-1].split(".")[0]
        pts = line_arr[:196]
        with Path(wflw_path + img + ".pts").open("w") as f:
            f.write("version: 1\n")
            f.write(f"n_points: {len(points_map)}\n")
            f.write("{\n")
            for j in range(0, len(pts), 2):
                if j // 2 in points_map:
                    f.write(f"{pts[j]} {pts[j + 1]}\n")
            f.write("}")

    print("done :)")


def reflect_datasets() -> None:
    """reflects the datasets horizontally to double the size"""
    # delete all files with "*_r_r*"
    # oops
    system("find /dcs/large/u2204489/eyes -type f -name '*_r_r.*' -exec rm -f {} +")
    system("find /dcs/large/u2204489/eyes -type f -name '*_r.*' -exec rm -f {} +")

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

    files = list(Path(path).glob("*.pts"))
    for file in files:
        print("Reflecting file:", file)
        # reflect the image
        # (thanks to python3.12 for embedded f-strings)
        img = cv.imread(f"{file.with_suffix(f".{type}")}")
        # reflected_img = cv.flip(img, 1)
        # cv.imwrite(str(file.with_name(file.stem + f"_r.{type}")), reflected_img)

        # reflect the points
        with file.open() as f:
            points = np.loadtxt(f, comments=("version:", "n_points:", "{", "}"))
        reflected_points = np.copy(points)
        reflected_points[:, 0] = img.shape[1] - reflected_points[:, 0]
        file_content = "version: 1\nn_points: 68\n{\n"
        for x, y in reflected_points:
            file_content += f"{x} {y}\n"
        file_content += "}"
        with file.with_name(file.stem + "_r.pts").open("w") as f:
            f.write(file_content)


if __name__ == "__main__":
    arg = None
    try:
        arg = sys.argv[1]
    except IndexError:
        print("invalid argument")
        sys.exit(1)

    if arg == "300w":
        check_300w_dataset()
    elif arg == "cofw":
        format_cofw_dataset()
    elif arg == "wflw":
        format_wflw_dataset()
    elif arg == "visualise":
        visualise_eye_datasets()
    elif arg == "reflect":
        reflect_datasets()
    else:
        print("invalid argument")
        sys.exit(1)
