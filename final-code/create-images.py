import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2 as cv
from mtcnn import MTCNN
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

def iou(box1: list[int], box2: list[int]) -> float:
    """calculates the intersection over union of two boxes"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # calculate the intersection area
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(x1 + w1, x2 + w2)
    inter_y2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    # calculate the union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # calculate the IoU
    return inter_area / union_area if union_area > 0 else 0

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

        yunet = cv.FaceDetectorYN().create(
            "face_detection_yunet_2023mar.onnx", "", (0,0), 0.5
        )
        mtcnn = MTCNN("face_detection_only", "CPU:0")

        # get n random frames from the video
        num_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        random.seed(42)
        frames = sorted(random.sample(range(num_frames), 32))

        # extract those n frames, resize, then save
        frame_num = 0
        previous_face = None
        for frame_id in frames:
            video.set(cv.CAP_PROP_POS_FRAMES, frame_id)
            success, frame = video.read()
            if not success:
                continue
            yunet.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = yunet.detect(frame)
            if len(faces) == 0:
                faces = mtcnn.detect_faces(frame)
                if faces is None:
                    continue
                faces = [face["box"] for face in faces]
            # choose face with highest iou with previous face
            if previous_face is not None:
                # choose face with highest iou with previous face
                faces = sorted(
                    faces,
                    key=lambda face: iou(previous_face, list(map(int, face[:4]))),
                    reverse=True,
                )
            else:
                # choose face with highest area
                faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)
            x, y, w, h = list(map(int, faces[0][:4]))
            frame = frame[y:y+h, x:x+w]
            frame = cv.resize(frame, frame_size)
            cv.imwrite(f"{video_path[:-4]}_{frame_num}.png", frame)
            frame_num += 1

    # process videos in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(process_video, dataset)
    print("Done :)")

if __name__ == "__main__":
    save_frames((256, 256), sys.argv[1])
