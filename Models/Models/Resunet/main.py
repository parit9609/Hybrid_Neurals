import numpy as np
import os
import pickle
import glob
import cv2
from imageio import imread, imsave
import imageio.v2 as imageio
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from sklearn import metrics
from tqdm import tqdm
from prepare_data import *
from data_process import *
from model1 import ResUNet  # Changed import statement
import imageio.v2 as imageio

# Define Jaccard loss function
class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, y_pred, y_true):
        eps = 1e-15
        intersection = torch.sum(y_pred * y_true)
        union = torch.sum(y_pred) + torch.sum(y_true) - intersection
        jac = (intersection + eps) / (union + eps)
        return 1 - jac

# Define evaluation functions
def Sens(y_true, y_pred):
    cm1 = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])  # labels =[1,0] [positive [Hemorrhage], negative]
    SensI = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    return SensI  # TPR is also known as sensitivity

def Speci(y_true, y_pred):
    cm1 = metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])  # labels =[1,0] [positive [Hemorrhage], negative]
    SpeciI = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    return SpeciI
    
def Jaccard_img(y_true, y_pred):
    iou_score = 0
    counter = 0
    for i in range(y_true.shape[0]):
        if torch.sum(torch.from_numpy(y_true[i])) > 0:  # Convert NumPy array to PyTorch tensor
            im1 = torch.from_numpy(y_true[i]).bool()  # Convert to PyTorch tensor and then to bool
            im2 = torch.from_numpy(y_pred[i]).bool()  # Convert to PyTorch tensor and then to bool
            intersection = torch.logical_and(im1, im2)
            union = torch.logical_or(im1, im2)
            iou_score += torch.sum(intersection).item() / torch.sum(union).item()
            counter += 1
    if counter > 0:
        return iou_score / counter
    else:
        return float('nan')

def dice_img(y_true, y_pred):
    dice = 0
    counter = 0
    for i in range(y_true.shape[0]):
        if torch.sum(torch.from_numpy(y_true[i])) > 0:  # Convert NumPy array to PyTorch tensor
            dice += dice_fun(y_true[i], y_pred[i])
            counter += 1
    if counter > 0:
        return dice / counter
    else:
        return float('nan')

def dice_fun(im1, im2):
    im1 = torch.from_numpy(im1).bool()  # Convert NumPy array to PyTorch tensor and then to boolean tensor
    im2 = torch.from_numpy(im2).bool()  # Convert NumPy array to PyTorch tensor and then to boolean tensor

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = torch.logical_and(im1, im2)

    return 2. * torch.sum(intersection).float() / (torch.sum(im1).float() + torch.sum(im2).float())

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, images_path, masks_path, transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        image = imageio.imread(self.images_path[idx])  # Updated this line
        mask = imageio.imread(self.masks_path[idx])
        
        # Convert single-channel image to three channels
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# Define train function
def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
            val_loss /= len(val_loader.dataset)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

# Main script
if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define parameters
    num_CV = 5
    NumEpochs = 100
    batch_size = 32
    learning_rate = 1e-5
    detectionSen = 20 * 20
    thresholdI = 0.5
    detectionThreshold = thresholdI * 256
    numSubj = 75
    imageLen = 512
    windowLen = 128
    strideLen = 64
    num_Moves = int(imageLen / strideLen) - 1
    window_specs = [40, 120]
    kernel_closing = torch.ones((1, 1, 10, 10))
    kernel_opening = torch.ones((1, 1, 5, 5))

    # Initialize directories
    counterI = 1
    SaveDir = Path('results_trial' + str(counterI))
    while os.path.isdir(str(SaveDir)):
        counterI += 1
        SaveDir = Path('results_trial' + str(counterI))
    os.mkdir(str(SaveDir))
    os.mkdir(str(Path(SaveDir, 'crops')))
    os.mkdir(str(Path(SaveDir, 'fullCT_original')))
    os.mkdir(str(Path(SaveDir, 'fullCT_morph' + str(thresholdI))))

    # Prepare data
    dataset_zip_dir = 'computed-tomography-images-for-intracranial-hemorrhage-detection-and-segmentation-1.3.1.zip'
    crossvalid_dir = 'DataV1'
    prepare_data(dataset_zip_dir, crossvalid_dir, numSubj, imageLen, windowLen, strideLen, num_Moves, window_specs)

    # Load data
    with open(str(Path(crossvalid_dir, 'ICH_DataSegmentV1.pkl')), 'rb') as Dataset1:
        [hemorrhageDiagnosisArray, AllCTscans, testMasks, subject_nums_shaffled] = pickle.load(Dataset1)
    del AllCTscans
    testMasks = np.uint8(testMasks)
    testMasksAvg = np.where(np.sum(np.sum(testMasks, axis=1), axis=1) > detectionSen, 1, 0)
    testPredictions = np.zeros((testMasks.shape[0], imageLen, imageLen), dtype=np.uint8)

    # Cross-validation loop
    print('Starting the cross-validation!!')
    for cvI in range(0, num_CV):
        print("Working on fold #" + str(cvI) + ", starting training U-Net")
        SaveDir_crops_cv = Path(SaveDir, 'crops', 'CV' + str(cvI))
        if not os.path.isdir(str(SaveDir_crops_cv)):
            os.makedirs(str(SaveDir_crops_cv))
        SaveDir_full_cv = Path(SaveDir, 'fullCT_original', 'CV' + str(cvI))
        if not os.path.isdir(str(SaveDir_full_cv)):
            os.makedirs(str(SaveDir_full_cv))
        SaveDir_cv = Path(SaveDir, 'fullCT_morph' + str(thresholdI), 'CV' + str(cvI))
        if not os.path.isdir(str(SaveDir_cv)):
            os.makedirs(str(SaveDir_cv))

        dataDir = Path(crossvalid_dir, 'CV' + str(cvI))
        images_train = glob.glob(os.path.join(str(Path(dataDir, 'train', 'image')), "*.png"))
        masks_train = glob.glob(os.path.join(str(Path(dataDir, 'train', 'label')), "*.png"))
        images_val = glob.glob(os.path.join(str(Path(dataDir, 'validate', 'image')), "*.png"))
        masks_val = glob.glob(os.path.join(str(Path(dataDir, 'validate', 'label')), "*.png"))
        images_test = glob.glob(os.path.join(str(Path(dataDir, 'test', 'crops', 'image')), "*.png"))

        train_dataset = CustomDataset(images_train, masks_train, transform=ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = CustomDataset(images_val, masks_val, transform=ToTensor())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model, criterion, optimizer
        model = ResUNet(n_channels=1, n_classes=1).to(device)  # Change model initialization to ResUNet
        criterion = JaccardLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        train_model(model, criterion, optimizer, train_loader, val_loader, NumEpochs)

       # Save the trained model
        model_save_path = Path(SaveDir, 'unet_CV' + str(cvI) + '.pth')
        torch.save(model.state_dict(), model_save_path)
        print("Trained model saved to:", model_save_path)

        # Testing the trained model
        print('Testing the best U-Net model on testing data and saving the results to:', str(SaveDir_crops_cv))
        # No need to reinitialize the model, use the trained model
        model.eval()
        # Loop through test images and make predictions
        for img_path in tqdm(images_test, desc="Testing"):
            try:
                img = imread(img_path, pilmode='RGB')
                img_tensor = ToTensor()(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    prediction = model(img_tensor)
                    prediction = torch.squeeze(prediction).cpu().numpy()
                    # Save the prediction
                    imsave(os.path.join(SaveDir_crops_cv, os.path.basename(img_path)), np.uint8(prediction * 255))
            except Exception as e:
                 print(f"Error processing image {img_path}: {e}")

        # Creating full image mask from the crops predictions
        if cvI < num_CV - 1:
            subjectNums_cvI_testing = subject_nums_shaffled[cvI * int(numSubj / num_CV):cvI * int(numSubj / num_CV) + int(numSubj / num_CV)]
        else:
            subjectNums_cvI_testing = subject_nums_shaffled[cvI * int(numSubj / num_CV):numSubj]

        print('Combining the crops masks to find the full CT mask after performing morphological operations and saving the results to:', str(SaveDir_full_cv))
        for subItest in range(0, len(subjectNums_cvI_testing)):
            slicenum_s = hemorrhageDiagnosisArray[hemorrhageDiagnosisArray[:, 0] == subjectNums_cvI_testing[subItest], 1]
            sliceInds = np.where(hemorrhageDiagnosisArray[:, 0] == subjectNums_cvI_testing[subItest])[0]
            counterSlice = 0
            for sliceI in range(slicenum_s.size):
                CTslicePredict = np.zeros((imageLen, imageLen))
                windowOcc = np.zeros((imageLen, imageLen))
                counterCrop = 0
                for i in range(num_Moves):
                    for j in range(num_Moves):
                        windowI = imageio.imread(Path(SaveDir_crops_cv, str(subjectNums_cvI_testing[subItest]) +  str(sliceI) + str(counterCrop) + '.png'))
                        windowI = windowI / 255
                        CTslicePredict[int(i * imageLen / (num_Moves + 1)):int(i * imageLen / (num_Moves + 1) + windowLen),
                            int(j * imageLen / (num_Moves + 1)):int(j * imageLen / (num_Moves + 1) + windowLen)] = CTslicePredict[int(i * imageLen / (num_Moves + 1)):int(i * imageLen / (num_Moves + 1) + windowLen),
                            int(j * imageLen / (num_Moves + 1)):int(j * imageLen / (num_Moves + 1) + windowLen)] + windowI
                        windowOcc[int(i * imageLen / (num_Moves + 1)):int(i * imageLen / (num_Moves + 1) + windowLen),
                            int(j * imageLen / (num_Moves + 1)):int(j * imageLen / (num_Moves + 1) + windowLen)] = windowOcc[int(i * imageLen / (num_Moves + 1)):int(i * imageLen / (num_Moves + 1) + windowLen),
                            int(j * imageLen / (num_Moves + 1)):int(j * imageLen / (num_Moves + 1) + windowLen)] + 1
                        counterCrop = counterCrop + 1
                CTslicePredict = CTslicePredict / windowOcc * 255
                img = np.uint8(CTslicePredict)
                imsave(Path(SaveDir_full_cv, str(subjectNums_cvI_testing[subItest])  + str(sliceI) + '.png'), img)

                img = np.int16(np.where(img > detectionThreshold, 255, 0))
                img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_closing.numpy())
                img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_opening.numpy())
                imsave(Path(SaveDir_cv, str(subjectNums_cvI_testing[subItest])  + str(sliceI) + '.png'), np.uint8(img))
                testPredictions[sliceInds[counterSlice]] = np.uint8(np.where(img > (0.5*256), 1, 0))
                counterSlice += 1

    # Post-processing and evaluation
    CVtestPredictionsAvg = np.where(np.sum(np.sum(testPredictions, axis=1), axis=1) > detectionSen, 1, 0)
    class_report = np.zeros((numSubj, 14))
    for subjI in range(numSubj):
        sliceInds = np.where(hemorrhageDiagnosisArray[:, 0] == subjI)[0]
        class_report[subjI, 0] = Jaccard_img(testMasks[sliceInds], testPredictions[sliceInds])
        class_report[subjI, 1] = dice_img(testMasks[sliceInds], testPredictions[sliceInds])
        class_report[subjI, 2] = metrics.accuracy_score(testMasksAvg, CVtestPredictionsAvg)
        class_report[subjI, 3] = metrics.recall_score(testMasksAvg, CVtestPredictionsAvg, pos_label=1)
        class_report[subjI, 4] = metrics.precision_score(testMasksAvg, CVtestPredictionsAvg, pos_label=1)
        class_report[subjI, 5] = metrics.f1_score(testMasksAvg, CVtestPredictionsAvg, pos_label=1)
        class_report[subjI, 6] = Sens(testMasksAvg, CVtestPredictionsAvg)
        class_report[subjI, 7] = Speci(testMasksAvg, CVtestPredictionsAvg)

    class_report[21, :] = np.nan
    print("Final pixel-wise testing: mean Jaccard %.3f (max %.3f, min %.3f, +- %.3f), mean Dice %.3f (max %.3f, min %.3f, +- %.3f)" % (
            np.nanmean(class_report[:, 0]), np.nanmax(class_report[:, 0]), np.nanmin(class_report[:, 0]),
            np.nanstd(class_report[:, 0]),
            np.nanmean(class_report[:, 1]), np.nanmax(class_report[:, 1]), np.nanmin(class_report[:, 1]), np.nanstd(class_report[:, 1])))
    print("Final testing: Accuracy %.3f (max %.3f, min %.3f, +- %.3f), Sensi %.4f (max %.3f, min %.3f, +- %.3f), Speci %.4f (max %.3f, min %.3f, +- %.3f))." % (
        np.nanmean(class_report[:, 2]), np.nanmax(class_report[:, 2]), np.nanmin(class_report[:, 2]),
        np.nanstd(class_report[:, 2])
        , np.nanmean(class_report[:, 3]), np.nanmax(class_report[:, 3]), np.nanmin(class_report[:, 3]), np.nanstd(class_report[:, 3])
        , np.nanmean(class_report[:, 3]), np.nanmax(class_report[:, 3]), np.nanmin(class_report[:, 3]), np.nanstd(class_report[:, 3])))

    
    with open(str(Path(SaveDir, 'fullCT_morph' + str(thresholdI), 'report.pkl')), 'wb') as Results:
        pickle.dump([class_report, testMasks, testPredictions], Results)
