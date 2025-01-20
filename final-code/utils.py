# utils for project

import sys
from pathlib import Path

import cv2 as cv  # noqa: F401
import numpy as np

path_to_eyes = "/dcs/large/u2204489/eyes"


def format_helen_dataset() -> None:
    """reformats helen landmarks from txt files to pts files
    <filename>.jpg
    <filename>.pts
    """
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
                    w.write("}\n")
        Path(f"{path_to_eyes}/helen/{i}.txt").unlink()


def format_eye_datasets(
    path: str,
    landmark_indices: list[list[int]],
    strong: bool,
) -> None:
    files = list(Path(path).glob("*.pts"))
    for file in files:
        points = np.loadtxt(file, comments=("version:", "n_points:", "{", "}"))

        file_content = ""
        # create bounding box
        for landmark in landmark_indices:
            landmarks = points[landmark]
            x, y, w, h = cv.boundingRect(landmarks)
            file_content += f"{x} {y} {w} {h}\n"
        if strong:
            eye_landmarks = points[landmark_indices[0]]
            for x, y in eye_landmarks:
                file_content += f"{x} {y}\n"
        # save to file
        with file.with_suffix(".txt").open("w") as f:
            f.write(file_content)
        file.unlink()


def visualise_points(path: str) -> None:
    with Path(f"{path}.pts").open() as f:
        points = np.loadtxt(f, comments=("version:", "n_points:", "{", "}"))
        img = cv.imread(f"{path}.jpg")
        for x, y in points:
            cv.circle(img, (int(float(x)), int(float(y))), 1, (0, 0, 255), -1)
            cv.putText(
                img,
                f"{x}, {y}",
                (int(float(x)), int(float(y))),
                cv.FONT_HERSHEY_SIMPLEX,
                0.3,
                (0, 0, 0),
                1,
            )
        cv.imshow("image", img)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    arg = None
    try:
        arg = sys.argv[1]
    except IndexError:
        print("invalid argument")
        sys.exit(1)

    if arg == "helen":
        format_helen_dataset()
