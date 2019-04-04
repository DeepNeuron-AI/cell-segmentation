import os

import cv2
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

from common.exceptions import CellDatasetException


class CellDataset(Dataset):
    """
    Returns dataset instance for data-science-bowl-2018.
    Args:
        image_folder (str): the folder containing the images.
        root_dir (str): directory path to dataset folder.
        transform (transforms.Compose): list of transformations to apply to
                                        input data.
    """

    def __init__(self, image_folder: str, root_dir: str, edges: bool = False, transform: transforms.Compose = None,
                 target_transform: transforms.Compose = None):

        # Initialise parameters
        self.image_folder = image_folder
        self.root_dir = root_dir
        self.edges = edges
        self.transform = transform
        self.target_transform = target_transform
        self.csv_file = "stage1_train_labels.csv"

        # pull information csv
        try:
            self.frame = pd.read_csv(os.path.join(root_dir, self.csv_file))
        except FileNotFoundError:
            raise CellDatasetException(
                "Could not read csv_file: %s\nEnsure root_dir points to dataset directory. \nDownload from "
                "https://www.kaggle.com/c/data-science-bowl-2018/data or with data_downloader.py" %
                os.path.join(root_dir, self.csv_file))

    def _load_mask(self, iloc):
        """
        Returns combined mask files.
        Args:
            iloc (int): index in the data frame.
        Returns:
            mask (PIL.image): mask image.
        """
        # Collect image name from csv frame
        img_name = self.frame.ImageId.unique()[iloc]
        mask_dir = os.path.join(self.root_dir, self.image_folder, img_name, "masks")
        mask_paths = [os.path.join(mask_dir, fp) for fp in os.listdir(mask_dir)]

        mask = None
        for fp in mask_paths:
            img = cv2.imread(fp, 0)
            if img is None:
                raise FileNotFoundError("Could not open %s" % fp)
            if self.edges:
                img = cv2.Canny(img, 1, 255, L2gradient=True)
            if mask is None:
                mask = img
            else:
                mask = np.maximum(mask, img)

        mask = Image.fromarray(mask)
        return mask

    def _get_image(self, iloc):
        """
        Gets an image from the directory
        Args:
            iloc (int): The position in the dataframe
        Returns:
            image (torch.tensor): The image in RGB
        """
        # Collect image name from csv frame
        img_name = self.frame.ImageId.unique()[iloc]
        img_path = os.path.join(
            self.root_dir, self.image_folder, img_name, "images", img_name + '.png')

        # Load image
        image = Image.open(img_path).convert('RGB')

        return image

    def __len__(self):
        return len(self.frame.ImageId.unique())

    def __getitem__(self, iloc):
        """
        Gets the image and details for training
        Args:
            iloc (int): The position in the dataframe
        Returns:
            image (torch.tensor): The image in RGB
            mask (torch.tensor): A mask of the cell locations
            number (int): The number of cells in the image
        """
        image = self._get_image(iloc)
        mask = self._load_mask(iloc)

        # invert if too bright
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, a = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        b = np.count_nonzero(a)
        ttl = np.prod(a.shape)
        if b > ttl / 2:
            image = Image.fromarray(cv2.bitwise_not(img))

        if self.transform:
            image = self.transform(image)
            mask = self.target_transform(mask)

        number = len(self.frame[self.frame.ImageId == self.frame.ImageId.unique()[iloc]])

        # from utils.utils import unnormalise
        # img = unnormalise(image)
        # msk = unnormalise(mask)
        # cv2.imshow("img", img)
        # cv2.imshow("msk", msk)
        # cv2.waitKey()

        return image, mask, number

    @property
    def shape(self):
        image, mask, _ = self.__getitem__(0)
        return image.shape, mask.shape, 1
