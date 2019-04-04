from torchvision import transforms as transforms

from data_loader.cell_dataset import CellDataset


def import_cell_dataset(dataset_root: str, train: bool = True, crop_size: int = 256, edges: bool = False):
    image_folder = "stage1_train" if train else "stage1_test"
    dataset = CellDataset(
        image_folder=image_folder,
        root_dir=dataset_root,
        edges=edges,
        transform=transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]),
        target_transform=transforms.Compose([
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
        ])
    )
    return dataset
