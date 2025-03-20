import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
from sklearn.model_selection import train_test_split


def generate_datasets(path_to_dataset: str) -> list[list[tuple[str, int]]]:
    """creates dataset of video paths and labels, splitting into train and test sets"""
    videos = list(Path(path_to_dataset).rglob("*.mp4"))

    real_videos = [str(video) for video in videos if "real" in str(video)]
    fake_videos = [str(video) for video in videos if "fake" in str(video)]

    train_size = int(0.8 * len(real_videos))

    train_real, test_real = train_test_split(
        real_videos, train_size=train_size, random_state=42
    )
    train_fake, test_fake = train_test_split(
        fake_videos, train_size=train_size, random_state=42
    )

    train_data = (
        [(video, 1) for video in train_real] + [(video, 0) for video in train_fake]
    )
    test_data = (
        [(video, 1) for video in test_real] + [(video, 0) for video in test_fake]
    )

    return [train_data, test_data]

def save_frames(
    frame_size: tuple[int, int],
    path_to_dataset: str,
) -> None:
    """Extract and save frames from videos in a dataset."""

    dataset = generate_datasets(path_to_dataset)[0]

    labels = ["fake", "real"]
    num_videos = [0, 0]
    max_videos = sum(label for _, label in dataset)

    def process_video(video_path: str, label: int) -> None:
        """Extract and save frames from a video."""
        nonlocal num_videos
        if num_videos[label] >= max_videos:
            return

        num_videos[label] += 1
        video = cv.VideoCapture(video_path)
        frame_num = 0

        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            frame = cv.resize(frame, frame_size)
            path = f"{path_to_dataset}/{labels[label]}/{num_videos[label]}_{frame_num}.jpg" # noqa: E501
            cv.imwrite(path, frame)
            frame_num += 1

        video.release()
        return

    # process videos in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(
            process_video,
            (video for video, _ in dataset),
            (label for _, label in dataset),
        )

if __name__ == "__main__":
    save_frames((256, 256), sys.argv[1])
