import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
from sklearn.model_selection import train_test_split


def generate_datasets(path_to_dataset: str) -> list[str]:
    """creates dataset of video paths"""

    # get all video paths
    videos = list(Path(path_to_dataset).rglob("*.mp4"))

    # split videos into real and fake
    real_videos = [str(video) for video in videos if "real" in str(video)]
    fake_videos = [str(video) for video in videos if "fake" in str(video)]

    # reals are the limiting factor
    train_size = int(len(real_videos) * 0.8)

    real_videos, _ = train_test_split(
        real_videos, train_size=train_size, random_state=42
    )
    fake_videos, _ = train_test_split(
        fake_videos, train_size=train_size, random_state=42
    )

    # combine the two lists
    return real_videos + fake_videos

def save_frames(
    frame_size: tuple[int, int],
    path_to_dataset: str,
) -> None:
    """Extract and save frames from videos in a dataset"""

    dataset = generate_datasets(path_to_dataset)

    def process_video(video_path: str) -> None:
        """Extract and save frames from a video."""
        print(f"Processing {video_path}")
        video = cv.VideoCapture(video_path)

        # get n random frames from the video
        num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        random.seed(42)
        frames = sorted(random.sample(range(num_frames), 100))

        # extract those n frames, resize, then save
        frame_num = 0
        for frame_id in frames:
            video.set(cv.CAP_PROP_POS_FRAMES, frame_id)
            success, frame = video.read()
            if not success:
                continue
            frame = cv.resize(frame, frame_size)
            cv.imwrite(f"{video_path[:-4]}_{frame_num}.jpg", frame)
            frame_num += 1

    # process videos in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(process_video, dataset)
    print("Done :)")

if __name__ == "__main__":
    save_frames((256, 256), sys.argv[1])
