import os
import pathlib
import shutil
from tqdm import tqdm

def change_directories(dataset_directory):
    dataset_path = pathlib.Path(dataset_directory)
    for scene in dataset_path.iterdir():
        scene_path = pathlib.Path(scene)
        for subject in scene_path.iterdir():
            subject_path = pathlib.Path(subject)
            for action in subject_path.iterdir():
                if action.is_dir():
                    index = action.name[-2:]
                    index = int(index) + 27
                    new_folder_name = f"{action.name[:-2]}{index}"
                    new_folder_path = action.parent / new_folder_name
                    action.rename(new_folder_path)


def copy_from_mmfi(dataset_directory, destination_directory):
    dataset_path = pathlib.Path(dataset_directory)
    for scene in dataset_path.iterdir():
        scene_path = pathlib.Path(scene)
        for subject in scene_path.iterdir():
            subject_path = pathlib.Path(subject)
            for action in tqdm(subject_path.iterdir()):
                action_path = pathlib.Path(action)
                for modality in action_path.iterdir():
                    if modality.is_dir() and modality.name == 'lidar':
                        ori_folder_path = modality.parent / modality.name
                        for root, dirs, files in os.walk(ori_folder_path):
                            for frame in files:
                                ori_lidar = os.path.join(root, frame)
                                lidar_path = ori_lidar.split(dataset_directory)[1]
                                new_lidar_path = f"{destination_directory}{lidar_path}"
                                new_lidar_folder = new_lidar_path.split(frame)[0]
                                if not os.path.exists(new_lidar_folder):
                                    os.makedirs(new_lidar_folder)
                                shutil.copy2(ori_lidar, new_lidar_path)
                    if modality.name == 'ground_truth.npy':
                        ori_gt = str(modality)
                        gt_path = ori_gt.split(dataset_directory)[1]
                        new_gt_path = f"{destination_directory}{gt_path}"
                        new_gt_folder = new_gt_path.split(modality.name)[0]
                        if not os.path.exists(new_gt_folder):
                            os.makedirs(new_gt_folder)
                        shutil.copy2(ori_gt, new_gt_path)

def copy_generated(source_directory, destination_directory):
    for root, dirs, files in os.walk(source_directory):
        relative_path = os.path.relpath(root, source_directory)
        target_path = os.path.join(destination_directory, relative_path)

        if not os.path.exists(target_path):
            os.makedirs(target_path)

        for file in tqdm(files):
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_path, file)
            shutil.copy2(source_file, target_file)


def main():
    origin_path = "D:\\AI\\mmfi"
    generated_path = "C:\\Users\\27309\\Desktop\\generated"
    destination_path = "C:\\Users\\27309\\Desktop\\merged"

    if not os.path.exists(destination_path):
        os.makedirs(destination_path)
    copy_generated(generated_path, destination_path)
    change_directories(destination_path)
    copy_from_mmfi(origin_path, destination_path)


if __name__ == '__main__':
    main()



