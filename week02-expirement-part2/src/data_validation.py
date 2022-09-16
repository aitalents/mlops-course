from config import AppConfig
from PIL import Image
from tqdm import tqdm


def main_actions(config: AppConfig):
    images_path_list = [x for x in config.dataset_path.glob("**/*.jpg")]
    for image_path in tqdm(images_path_list, desc="Images validation"):
        Image.open(image_path)


def main():
    config = AppConfig.parse_raw()
    main_actions(config=config)


if __name__ == "__main__":
    main()