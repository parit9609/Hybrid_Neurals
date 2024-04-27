import numpy as np
import os
import glob
import skimage.transform as trans
from imageio import imsave, imread
from pathlib import Path
from torchvision import transforms
import torchvision

def adjustData(img, mask, flag_multi_class, num_class):
    if flag_multi_class:
        img = img / 255
        mask = mask[:, :, :, 0] if len(mask.shape) == 4 else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i, i] = 1
        new_mask = np.reshape(new_mask, (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2], new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask, (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask
    elif np.max(img) > 1:
        img = img / 255
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return img, mask

def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                    flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(128, 128), seed=1):
    image_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    image_generator = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=image_transforms
    )
    
    mask_generator = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=mask_transforms
    )

    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        img = np.array(img)
        mask = np.array(mask)
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)

def validateGenerator(batch_size, train_path, image_folder, mask_folder, image_color_mode="grayscale",
                    mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                    flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(128, 128), seed=1):
    image_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    mask_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    image_generator = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=image_transforms
    )
    
    mask_generator = torchvision.datasets.ImageFolder(
        root=train_path,
        transform=mask_transforms
    )

    validate_generator = zip(image_generator, mask_generator)
    for (img, mask) in validate_generator:
        img = np.array(img)
        mask = np.array(mask)
        img, mask = adjustData(img, mask, flag_multi_class, num_class)
        yield (img, mask)

def testGenerator(test_path, target_size=(128, 128), flag_multi_class=False, as_gray=True):
    image_transforms = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    image_dataset = torchvision.datasets.ImageFolder(
        root=test_path,
        transform=image_transforms
    )

    for img, _ in image_dataset:
        img = np.array(img)
        img = img / 255
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if not flag_multi_class else img
        img = np.reshape(img, (1,) + img.shape)
        yield img

def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = imread(item)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix))
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def saveResult(test_path,save_path,npyfile,flag_multi_class = False,num_class = 2):
    image_name_arr = glob.glob(os.path.join(test_path, "*.png"))
    string = image_name_arr[0];word = 'image'
    index = string.find(word)+6
    for i,item in enumerate(image_name_arr):
        img = npyfile[i,:,:,0]
        imsave(Path(save_path, item[index:]), np.uint8(img*255)) #index: to take only the image name that was read before

def LoadTestMask(test_path,num_image,target_size = (128,128),flag_multi_class = False,as_gray = True):
    Allsegment = np.zeros([num_image, target_size[0], target_size[1], 1], dtype=np.float32)
    counterI=0
    image_name_arr = glob.glob(os.path.join(test_path, "*.png"))
    for index, item in enumerate(image_name_arr):
        mask = imread(item)
        mask = mask / 255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        mask = trans.resize(mask, target_size)
        mask= np.expand_dims(mask, axis=-1)
        mask = np.expand_dims(mask, axis=0)
        Allsegment[counterI]=mask
        counterI = counterI + 1
    return Allsegment
