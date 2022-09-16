from config import AppConfig
from PIL import Image
from tqdm import tqdm


def main_actions(config: AppConfig):
    images_path_list = [x for x in config.dataset_path.glob("**/*.jpg")]
    dataset_path = config.dataset_output_path
    dataset_path.mkdir(parents=True, exist_ok=True)

    for image_path in tqdm(images_path_list, desc="Images validation"):
        class_name = image_path.parts[-2]  # Folder Name
        stage_name = image_path.parts[-3]  # Train/Test/Val
        class_folder = dataset_path/stage_name/class_name
        class_folder.mkdir(parents=True, exist_ok=True)
        Image.open(image_path).resize(size=(512, 512)).save(
            class_folder/image_path.name)


def main():
    config = AppConfig.parse_raw()
    main_actions(config=config)


if __name__ == "__main__":
    main()
