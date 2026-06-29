import os
import argparse
from tqdm import tqdm
from typing import List, Dict
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T

from networks.densenet_mcf import dense121_mcs
from dataloader.EyeQ_loader import InfrenceDataSet


def get_paths_to_imgs(base_dir: str, file_extension: str) -> List[str]:
    img_paths = []
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith(file_extension):
                img_paths.append(os.path.join(root, filename))
    return img_paths


def infere_img_labels(img_paths: str, batch_size: int, model_save_path: str) -> Dict[str, np.ndarray]:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')

    pre_transform = T.Compose([T.Resize(224), T.CenterCrop(224)])
    post_transform = T.Compose([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    dataset = InfrenceDataSet(img_paths, pre_transform, post_transform)
    dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = dense121_mcs(n_class=3)
    loaded_model = torch.load(model_save_path)
    model.load_state_dict(loaded_model['state_dict'])
    model = model.to(device).eval()

    output_metric = {}
    for img_paths, (imagesA, imagesB, imagesC) in tqdm(dataloader):
        _, _, _, _, result_mcs = model(imagesA.to(device), imagesB.to(device), imagesC.to(device))
        for img_path, result in zip(img_paths, result_mcs.detach().cpu().numpy()):
            # Normalize results to add up to 1
            output_metric[img_path] = result / result.sum()
    return output_metric


def infere_img_quality(base_dir: str, path_to_save_file: str, model_save_path: str, file_extension: str,
                       batch_size: int) -> None:
    img_paths = get_paths_to_imgs(base_dir, file_extension)
    labels_dict = infere_img_labels(img_paths, batch_size, model_save_path)
    labels_df = pd.DataFrame(data=labels_dict.values(), index=labels_dict.keys(), columns=['Good', 'Usable', 'Reject'])
    labels_df.index.name = 'Path'
    labels_df.to_csv(path_to_save_file, sep='\t')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Script to get image labels')
    parser.add_argument('--base_dir', help='Directory where all images are saved', type=str, required=True)
    parser.add_argument('--model_save_path', help='Where pretrained model is stored. Available on Github.', type=str,
                        required=True)
    parser.add_argument('--path_to_save_file', help='Where to store the results (tsv)', type=str, required=True)
    parser.add_argument('--file_extension', help='Define stored file format', type=str, required=False, default='.png')
    parser.add_argument('--batch_size', help='How many images to process at once.', type=int, required=False,
                        default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    infere_img_quality(**vars(args))
