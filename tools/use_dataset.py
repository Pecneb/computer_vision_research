from trajectorynet.utility.dataset import load_dataset

def main():
    dataset = load_dataset("./research_data/short_test_videos/rouen_video.joblib")
    print(dataset)


if __name__ == "__main__":
    main()