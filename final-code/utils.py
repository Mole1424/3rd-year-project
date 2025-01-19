# utils for project

import sys
from pathlib import Path

import cv2 as cv  # noqa: F401

path_to_eyes = "/dcs/large/u2204489/eyes"


def format_helen_dataset() -> None:
    """formats the helen dataset into the following form
    <filename>.jpg
    <filename>.pts
    """
    num_files = 2330

    # # landmarks are lines 1-194
    # landmarks = list(range(1, 195))
    # # preview on image 1 to see the landmarks
    # with Path(f"{path_to_eyes}/helen/1.txt").open() as f:
    #     lines = f.read().split("\n")
    #     image_title = lines[0]
    #     image = cv.imread(f"{path_to_eyes}/helen/{image_title}.jpg")

    #     for landmark in landmarks:
    #         x, y = lines[landmark].split(" , ")
    #         x, y = int(float(x)), int(float(y))
    #         cv.circle(
    #             image,
    #             (x, y),
    #             radius=3,
    #             color=(0, 0, 255),
    #             thickness=-1,
    #         )
    #         cv.putText(
    #             image,
    #             str(landmark),
    #             (x, y),
    #             cv.FONT_HERSHEY_SIMPLEX,
    #             0.3,
    #             (255, 0, 0),
    #             1,
    #             cv.LINE_AA,
    #         )
    #     cv.imshow("image", image)
    #     cv.waitKey(0)

    eye_landmarks = [146, 143, 138, 135, 152, 149, 126, 123, 118, 115, 132, 129]

    # iterate over all txt files and save the landmarks to pts files
    for i in range(1, num_files + 1):
        with Path(f"{path_to_eyes}/helen/{i}.txt").open() as f:
            lines = f.read().split("\n")
            image_title = lines[0]

            # save landmarks to pts file
            with Path(f"{path_to_eyes}/helen/{image_title}.pts").open("w") as w:
                w.write("version: 1\n")
                w.write(f"n_points: {len(eye_landmarks)}\n")
                w.write("{\n")
                for landmark in eye_landmarks:
                    x, y = lines[landmark].split(" , ")
                    w.write(f"{x} {y}\n")
                w.write("}\n")
        Path(f"{path_to_eyes}/helen/{i}.txt").unlink()


if __name__ == "__main__":
    arg = None
    try:
        arg = sys.argv[1]
    except IndexError:
        print("invalid argument")
        sys.exit(1)

    if arg == "helen":
        format_helen_dataset()
