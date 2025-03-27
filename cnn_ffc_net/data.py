import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class TransformNeRFBilinear(object):
    def __init__(self, img_size=(64, 64)):
        self.img_size = img_size

    def __call__(self, img):
        return transforms.Resize(self.img_size)(img)


class DatasetNeRF(Dataset):
    def __init__(self, dataset_path, size_transform=None, image_transform=None, label_transform=None, neg_label_transform=None) -> None:
        self.dataset_path = dataset_path
        self.images_path = os.path.join(dataset_path, 'images')
        self.labels_path = os.path.join(dataset_path, 'mirror_views')
        self.neg_labels_path = os.path.join(dataset_path, '180_views')

        self.image_files = sorted(os.listdir(self.images_path))
        self.label_files = sorted(os.listdir(self.labels_path))
        self.neg_label_files = sorted(os.listdir(self.neg_labels_path))

        self.size_transform = size_transform
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.neg_label_transform = neg_label_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_filename = self.image_files[idx]
        label_filename = self.label_files[idx]
        neg_label_filename = self.neg_label_files[idx]

        image_path = os.path.join(self.images_path, image_filename)
        img = Image.open(image_path).convert('RGB')

        label_path = os.path.join(self.labels_path, label_filename)
        label_img = Image.open(label_path).convert('RGB')

        neg_label_path = os.path.join(self.neg_labels_path, neg_label_filename)
        neg_label_img = Image.open(neg_label_path).convert('RGB')

        if self.size_transform:
            img = self.size_transform(img)
            label_img = self.size_transform(label_img)
            neg_label_img = self.size_transform(neg_label_img)

        if self.image_transform:
            img = self.image_transform(img)

        if self.label_transform:
            label_img = self.label_transform(label_img)

        if self.neg_label_transform:
            neg_label_img = self.neg_label_transform(neg_label_img)

        return img, label_img, neg_label_img, image_filename, label_filename, neg_label_filename


def get_data(data_path, img_size, batch_size, val_batch_size=10):
    train_dataset_path = os.path.join(data_path, "train")
    val_dataset_path = os.path.join(data_path, "val")
    test_dataset_path = os.path.join(data_path, "test")

    size_transform = TransformNeRFBilinear(img_size)
    image_transform = transforms.Compose([transforms.ToTensor()])
    label_transform = transforms.Compose([transforms.ToTensor()])
    neg_label_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = DatasetNeRF(train_dataset_path, size_transform=size_transform,  image_transform=image_transform, label_transform=label_transform, neg_label_transform=neg_label_transform)
    val_dataset = DatasetNeRF(val_dataset_path, size_transform=size_transform, image_transform=image_transform, label_transform=label_transform, neg_label_transform=neg_label_transform)
    test_dataset = DatasetNeRF(test_dataset_path, size_transform=size_transform, image_transform=image_transform, label_transform=label_transform, neg_label_transform=neg_label_transform)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    return trainloader, valloader, testloader, train_dataset, val_dataset, test_dataset
