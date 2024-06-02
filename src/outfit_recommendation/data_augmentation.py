from torchvision.transforms import v2
import torchvision.transforms as T
import torch


def get_data_augmentation_transforms():
    return {
        'train': v2.Compose([
            v2.PILToTensor(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.RandomHorizontalFlip(),
            v2.RandomPerspective(fill=255),
            v2.RandomAffine(30, fill=255),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        'val': v2.Compose([
            v2.PILToTensor(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
        'test': v2.Compose([
            v2.PILToTensor(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224),
            v2.ConvertImageDtype(torch.float32),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]),
    }
