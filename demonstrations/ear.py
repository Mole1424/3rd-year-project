import pickle
from pathlib import Path

from matplotlib import pyplot as plt


def main() -> None:
    path_to_data = "/dcs/large/u2204489/celeb-df_hrnet_ears_dataset.pkl"
    with Path(path_to_data).open("rb") as f:
        data = pickle.load(f)
    data = data[0] # only use training data
    for i, (ears, _) in enumerate(data):
        # only want real data
        # unpadded
        if any(ear == -1 for ear in ears):
            continue

        # plot ears
        plt.figure()
        plt.plot(ears)
        plt.title("Plot of EAR over frame")
        plt.xlabel("Frame")
        plt.ylabel("EAR")
        plt.savefig(f"images/ear_{i}.png")
        plt.close()


if __name__ == "__main__":
    main()
