# Modifications:
#   support aug in testing
""" Transforms Factory
Factory methods for building image transforms for use with TIMM (PyTorch Image Models)

Hacked together by / Copyright 2020 Ross Wightman
"""
import math

import torch
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, DEFAULT_CROP_PCT
from timm.data.auto_augment import rand_augment_transform, augment_and_mix_transform, auto_augment_transform
from timm.data.transforms import str_to_interp_mode, str_to_pil_interp, RandomResizedCropAndInterpolation,\
    ResizeKeepRatio, CenterCropOrPad, ToNumpy
from timm.data.random_erasing import RandomErasing
from timm.data.transforms_factory import transforms_noaug_train, transforms_imagenet_eval

from PIL import ImageFilter, ImageOps
import random


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.1, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)))
        return img


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class GrayScale(object):
    """
    Apply RGB to Gray to the PIL image.
    """

    def __init__(self, p=0.2):
        self.p = p
        self.transf = transforms.Grayscale(3)

    def __call__(self, img):
        if random.random() < self.p:
            return self.transf(img)
        else:
            return img


def transforms_imagenet_train_v2(
    img_size=224,
    scale=None,
    ratio=None,
    hflip=0.5,
    vflip=0.,
    color_jitter=0.4,
    auto_augment=None,
    interpolation='random',
    use_prefetcher=False,
    mean=IMAGENET_DEFAULT_MEAN,
    std=IMAGENET_DEFAULT_STD,
    re_prob=0.,
    re_mode='const',
    re_count=1,
    re_num_splits=0,
    separate=False,
    force_color_jitter=False,
    use_simple_random_crop=False,
    use_three_augment_ssl=False,
):
    """
    If separate==True, the transforms are returned as a tuple of 3 separate transforms
    for use in a mixing dataset that passes
     * all data through the first (primary) transform, called the 'clean' data
     * a portion of the data through the secondary transform
     * normalizes and converts the branches above with the third, final transform
    """
    if use_three_augment_ssl:
        assert auto_augment is None, f'three_aug and auto_aug conflict'

    scale = tuple(scale or (0.08, 1.0))  # default imagenet scale range
    ratio = tuple(ratio or (3. / 4., 4. / 3.))  # default imagenet ratio range
    if use_simple_random_crop:
        if interpolation == 'random':
            interpolation = (str_to_interp_mode('bilinear'),
                             str_to_interp_mode('bicubic'))
        else:
            interpolation = str_to_interp_mode(interpolation)

        primary_tfl = [
            transforms.Resize(img_size, interpolation=interpolation),
            transforms.RandomCrop(img_size, padding=4, padding_mode='reflect'),
        ]
    else:
        primary_tfl = [
            RandomResizedCropAndInterpolation(img_size,
                                              scale=scale,
                                              ratio=ratio,
                                              interpolation=interpolation)
        ]
    if hflip > 0.:
        primary_tfl += [transforms.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.:
        primary_tfl += [transforms.RandomVerticalFlip(p=vflip)]

    secondary_tfl = []
    disable_color_jitter = False
    if auto_augment:
        assert isinstance(auto_augment, str)
        # color jitter is typically disabled if AA/RA on,
        # this allows override without breaking old hparm cfgs
        disable_color_jitter = not (force_color_jitter or '3a' in auto_augment)
        if isinstance(img_size, (tuple, list)):
            img_size_min = min(img_size)
        else:
            img_size_min = img_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)
        if auto_augment.startswith('rand'):
            secondary_tfl += [rand_augment_transform(auto_augment, aa_params)]
        elif auto_augment.startswith('augmix'):
            aa_params['translate_pct'] = 0.3
            secondary_tfl += [
                augment_and_mix_transform(auto_augment, aa_params)
            ]
        else:
            secondary_tfl += [auto_augment_transform(auto_augment, aa_params)]
    elif use_three_augment_ssl:
        disable_color_jitter = True
        secondary_tfl += [
            transforms.RandomChoice(
                [GrayScale(p=1.0),
                 Solarization(p=1.0),
                 GaussianBlur(p=1.0)])
        ]

    if auto_augment is None and not disable_color_jitter:
        # color jitter is enabled when not using AA or when forced
        if isinstance(color_jitter, (list, tuple)):
            # color jitter should be a 3-tuple/list if spec brightness/contrast/saturation
            # or 4 if also augmenting hue
            assert len(color_jitter) in (3, 4)
        else:
            # if it's a scalar, duplicate for brightness, contrast, and saturation, no hue
            color_jitter = (float(color_jitter), ) * 3
        secondary_tfl += [transforms.ColorJitter(*color_jitter)]

    final_tfl = []
    if use_prefetcher:
        # prefetcher and collate will handle tensor conversion and norm
        final_tfl += [ToNumpy()]
    else:
        final_tfl += [
            transforms.ToTensor(),
            transforms.Normalize(mean=torch.tensor(mean),
                                 std=torch.tensor(std))
        ]
        if re_prob > 0.:
            final_tfl.append(
                RandomErasing(re_prob,
                              mode=re_mode,
                              max_count=re_count,
                              num_splits=re_num_splits,
                              device='cpu'))

    if separate:
        return transforms.Compose(primary_tfl), transforms.Compose(
            secondary_tfl), transforms.Compose(final_tfl)
    else:
        return transforms.Compose(primary_tfl + secondary_tfl + final_tfl)


def create_transform_v2(input_size,
                        is_training=False,
                        use_prefetcher=False,
                        no_aug=False,
                        scale=None,
                        ratio=None,
                        hflip=0.5,
                        vflip=0.,
                        color_jitter=0.4,
                        auto_augment=None,
                        interpolation='bilinear',
                        mean=IMAGENET_DEFAULT_MEAN,
                        std=IMAGENET_DEFAULT_STD,
                        re_prob=0.,
                        re_mode='const',
                        re_count=1,
                        re_num_splits=0,
                        crop_pct=None,
                        crop_mode=None,
                        tf_preprocessing=False,
                        separate=False,
                        use_aug_in_test=False,
                        use_simple_random_crop=False,
                        use_three_augment_ssl=False):

    if isinstance(input_size, (tuple, list)):
        img_size = input_size[-2:]
    else:
        img_size = input_size

    if tf_preprocessing and use_prefetcher:
        assert not separate, "Separate transforms not supported for TF preprocessing"
        from timm.data.tf_preprocessing import TfPreprocessTransform
        transform = TfPreprocessTransform(is_training=is_training,
                                          size=img_size,
                                          interpolation=interpolation)
    else:
        if is_training and no_aug:
            assert not separate, "Cannot perform split augmentation with no_aug"
            transform = transforms_noaug_train(img_size,
                                               interpolation=interpolation,
                                               use_prefetcher=use_prefetcher,
                                               mean=mean,
                                               std=std)
        elif is_training:
            transform = transforms_imagenet_train_v2(
                img_size,
                scale=scale,
                ratio=ratio,
                hflip=hflip,
                vflip=vflip,
                color_jitter=color_jitter,
                auto_augment=auto_augment,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                re_prob=re_prob,
                re_mode=re_mode,
                re_count=re_count,
                re_num_splits=re_num_splits,
                separate=separate,
                use_simple_random_crop=use_simple_random_crop,
                use_three_augment_ssl=use_three_augment_ssl)

        else:
            assert not separate, "Separate transforms not supported for validation preprocessing"
            transform = transforms_imagenet_eval(
                img_size,
                interpolation=interpolation,
                use_prefetcher=use_prefetcher,
                mean=mean,
                std=std,
                crop_pct=crop_pct,
                crop_mode=crop_mode)

    return transform
