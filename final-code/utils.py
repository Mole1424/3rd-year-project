# utils for project

import json
import pickle
import sys
from os import system
from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

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
    visualise_points(f"{path_to_eyes}/300w/indoor_001_r", "png")
    # afw
    visualise_points(f"{path_to_eyes}/afw/134212_1", "jpg")
    visualise_points(f"{path_to_eyes}/afw/134212_1_r", "jpg")
    # cofw
    visualise_points(f"{path_to_eyes}/cofw/61", "jpg")
    visualise_points(f"{path_to_eyes}/cofw/61_r", "jpg")
    # helen
    visualise_points(f"{path_to_eyes}/helen/12799337_1", "jpg")
    visualise_points(f"{path_to_eyes}/helen/12799337_1_r", "jpg")
    # ibug
    visualise_points(f"{path_to_eyes}/ibug/image_003_1", "jpg")
    visualise_points(f"{path_to_eyes}/ibug/image_003_1_r", "jpg")
    # lfpw
    visualise_points(f"{path_to_eyes}/lfpw/trainset/image_0001", "png")
    visualise_points(f"{path_to_eyes}/lfpw/trainset/image_0001_r", "png")
    # wflw
    visualise_points(f"{path_to_eyes}/wflw/0_Parade_marchingband_1_116", "jpg")
    visualise_points(f"{path_to_eyes}/wflw/0_Parade_marchingband_1_116_r", "jpg")


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
    print("Deleting all files with *_r_r.*")
    system("find /dcs/large/u2204489/eyes -type f -name '*_r_r.*' -exec rm -f {} +")
    print("Deleting all files with *_r.*")
    system("find /dcs/large/u2204489/eyes -type f -name '*_r.*' -exec rm -f {} +")
    print("Deleting files")

    reflect_dataset(f"{path_to_eyes}/300w/", "png")
    reflect_dataset(f"{path_to_eyes}/afw/", "jpg")
    reflect_dataset(f"{path_to_eyes}/cofw/", "jpg")
    reflect_dataset(f"{path_to_eyes}/helen/", "jpg")
    reflect_dataset(f"{path_to_eyes}/ibug/", "jpg")
    reflect_dataset(f"{path_to_eyes}/lfpw/testset/", "png")
    reflect_dataset(f"{path_to_eyes}/lfpw/trainset/", "png")
    reflect_dataset(f"{path_to_eyes}/wflw/", "jpg")

    print("done :)")


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
        reflected_img = cv.flip(img, 1)
        cv.imwrite(str(file.with_name(file.stem + f"_r.{type}")), reflected_img)

        # reflect the points
        with file.open() as f:
            points = np.loadtxt(f, comments=("version:", "n_points:", "{", "}"))
        file_content = "version: 1\nn_points: 68\n{\n"
        for x, y in points:
            file_content += f"{img.shape[1] - x} {y}\n"
        file_content += "}"
        with file.with_name(file.stem + "_r.pts").open("w") as f:
            f.write(file_content)


def correct_reflections() -> None:
    """corrects the reflections of the datasets"""
    correct_reflection(f"{path_to_eyes}/300w/")
    correct_reflection(f"{path_to_eyes}/afw/")
    correct_reflection(f"{path_to_eyes}/cofw/")
    correct_reflection(f"{path_to_eyes}/helen/")
    correct_reflection(f"{path_to_eyes}/ibug/")
    correct_reflection(f"{path_to_eyes}/lfpw/testset/")
    correct_reflection(f"{path_to_eyes}/lfpw/trainset/")
    correct_reflection(f"{path_to_eyes}/wflw/")
    print("done :)")


def correct_reflection(path: str) -> None:
    """corrects the reflections of an individual dataset"""
    # fmt: off
    # mapping from incorrect reflection to correct points
    mapping = [17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
               27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
               28, 29, 30, 31, 36, 35, 34, 33, 32,
               46, 45, 44, 43, 48, 47,
               40, 39, 38, 37, 41, 42,
               55, 54, 53, 52, 51, 50, 49, 60, 59, 58, 57, 56,
               65, 64, 63, 62, 61, 68, 67, 66]
    mapping = [x - 1 for x in mapping]
    # fmt: onsw
    files = list(Path(path).glob("*_r.pts"))
    files = files[1:]
    for file in files:
        print("Correcting file:", file)
        with file.open() as f:
            points = np.loadtxt(f, comments=("version:", "n_points:", "{", "}"))
        points = points[mapping]
        file_content = "version: 1\nn_points: 68\n{\n"
        for x, y in points:
            file_content += f"{x} {y}\n"
        file_content += "}"
        with file.open("w") as f:
            f.write(file_content)


def truncate_datasets() -> None:
    """truncate all datasets to 3dp"""
    truncate_dataset(f"{path_to_eyes}/300w/")
    truncate_dataset(f"{path_to_eyes}/afw/")
    truncate_dataset(f"{path_to_eyes}/cofw/")
    truncate_dataset(f"{path_to_eyes}/helen/")
    truncate_dataset(f"{path_to_eyes}/ibug/")
    truncate_dataset(f"{path_to_eyes}/lfpw/testset/")
    truncate_dataset(f"{path_to_eyes}/lfpw/trainset/")
    truncate_dataset(f"{path_to_eyes}/wflw/")
    print("done :)")


def truncate_dataset(path: str) -> None:
    """truncate reflecting points file to 3dp remove floating point errors"""
    files = list(Path(path).glob("*.pts"))
    for file in files:
        print("Truncating file:", file)
        with file.open() as f:
            points = np.loadtxt(f, comments=("version:", "n_points:", "{", "}"))
        file_content = "version: 1\nn_points: 68\n{\n"
        for x, y in points:
            file_content += f"{x:.3f} {y:.3f}\n"
        file_content += "}"
        with file.open("w") as f:
            f.write(file_content)


def move_files() -> None:
    """collates fake images into one directory"""
    path_to_fake = "/dcs/large/u2204489/faceforensics/fake/"

    fake_paths = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    fake_prefix_path = "/dcs/large/u2204489/faceforensics/manipulated_sequences/"
    fake_suffix_path = "/c23/videos/"

    for fake_path in fake_paths:
        print(f"Getting {fake_path}...")
        system(f"python ../datasets/faceforensics_download_v4.py -d {fake_path} --server EU2 -c c23 /dcs/large/u2204489/faceforensics")  # noqa: E501
        print(f"Moving {fake_path}...")
        complete_fake_path = fake_prefix_path + fake_path + fake_suffix_path
        files = list(Path(complete_fake_path).glob("*.mp4"))

        for file in files:
            file_name = fake_path.lower() + file.name
            system(f"mv {file} {path_to_fake + file_name}")

def reformat_dics(paths: list[str]) -> None:
    """corrects model order in dictionary"""

    desired_order = [
        "custom", "vgg", "resnet", "xception", "efficientnet",
        "custom_vgg", "vgg_vgg", "vgg_resnet", "vgg_xception", "vgg_efficientnet",
        "custom_resnet", "resnet_vgg", "resnet_resnet", "resnet_xception", "resnet_efficientnet",  # noqa: E501
        "custom_xception", "xception_vgg", "xception_resnet", "xception_xception", "xception_efficientnet",  # noqa: E501
        "custom_efficientnet", "efficientnet_vgg", "efficientnet_resnet", "efficientnet_xception", "efficientnet_efficientnet" # noqa: E501
    ]
    current_order = [
        "custom", "vgg", "resnet", "xception", "efficientnet",
        "custom_vgg", "custom_resnet", "custom_xception", "custom_efficientnet",
        "vgg_vgg", "vgg_resnet", "vgg_xception", "vgg_efficientnet",
        "resnet_vgg", "resnet_resnet", "resnet_xception", "resnet_efficientnet",
        "xception_vgg", "xception_resnet", "xception_xception", "xception_efficientnet",
        "efficientnet_vgg", "efficientnet_resnet", "efficientnet_xception", "efficientnet_efficientnet" # noqa: E501
    ]
    for path in paths:
        # load file
        with Path(path).open() as f:
            ear_analyser = f.readline().strip()
            results = json.load(f)
        # create a new dictionary with the desired order
        new_results = {}
        for old_key, new_key in zip(current_order, desired_order):
            if old_key in results:
                new_results[new_key] = results[old_key]
        # order dictionary by current order
        ordered_results = {}
        for key in current_order:
            ordered_results[key] = new_results[key]
        # save to file
        with Path(path).open("w") as f:
            f.write(ear_analyser + "\n")
            json.dump(ordered_results, f, indent=4)
            f.write("\n")
        print(f"{path} done :)")

def visualise_ear(path_to_ear: str) -> None:
    """visualises the ear dataset on training videos"""
    with Path(path_to_ear).open("rb") as f:
        ear_dataset = pickle.load(f)
    ear_dataset = ear_dataset[0] + ear_dataset[1]
    for i, (ear, label) in enumerate(ear_dataset):
        print(f"Visualising ear {i} - {label}")
        plt.figure()
        plt.plot(ear)
        plt.title(f"Label: {label}")
        plt.xlabel("Time (frames)")
        plt.ylabel("EAR")
        plt.savefig(f"ears/ear_{i}.png")
        plt.close()

def move(dataset_path: str) -> None:
    """flattens a dataset"""
    root = Path(dataset_path)

    for path in root.rglob("*"):
        if path.is_file():
            relative_path = path.relative_to(root)
            new_path = root / "-".join(relative_path.parts)
            print(f"Moving {path} to {new_path}")
            path.rename(new_path)

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
    elif arg == "correct":
        correct_reflections()
    elif arg == "truncate":
        truncate_datasets()
    elif arg == "move":
        move_files()
    elif arg == "reformat":
        paths = sys.argv[2:]
        reformat_dics(paths)
    elif arg == "ear":
        path_to_ear = sys.argv[2]
        visualise_ear(path_to_ear)
    elif arg == "move_dataset":
        dataset_path = sys.argv[2]
        move(dataset_path)
    else:
        print("invalid argument")
        sys.exit(1)
