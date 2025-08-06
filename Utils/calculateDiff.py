# coding=utf-8
import os
import numpy as np
import torch
from PIL import Image
import lpips
import tqdm
from skimage import metrics
import shutil


def calculate_threshold_accuracy(di, di_pred, thr):
    # Calculate max(di/di_pred, di_pred/di) and check if it's less than the threshold
    max_ratios = np.maximum(di / di_pred, di_pred / di)
    return np.mean(max_ratios < thr)


def calculate_single(target_image, output_image, lpips_model=lpips.LPIPS(net='alex')):
    target_image = np.array(target_image.squeeze().squeeze())
    output_image = np.array(output_image.squeeze().squeeze())
    target_image = ((target_image - target_image.min()) / (target_image.max() - target_image.min()) * 255)
    if output_image.max() == 0:
        output_image = np.zeros_like(target_image)
        print(f"Warning: output_image is all zeros, setting to zeros with shape {output_image.shape}")
    else:
        output_image = ((output_image - output_image.min()) / (output_image.max() - output_image.min()) * 255)
    target_image = np.rint(target_image)
    output_image = np.rint(output_image)
    diff_array = (target_image - output_image).astype(np.float32)
    diff_array[diff_array == 0] = 1e-6

    lossDict = dict()
    lossDict['mae'] = np.mean(np.abs(diff_array))
    lossDict['mse'] = np.mean(np.square(diff_array))
    lossDict['rmse'] = np.sqrt(np.mean(np.square(diff_array)))
    lossDict['acc1'] = calculate_threshold_accuracy(target_image, output_image, 1.25)
    lossDict['acc2'] = calculate_threshold_accuracy(target_image, output_image, 1.25 ** 2)
    lossDict['acc3'] = calculate_threshold_accuracy(target_image, output_image, 1.25 ** 3)

    lossDict['psnr'] = 10 * np.log10(65025 / lossDict['mse']) if lossDict['mse'] != 0 else float('inf')
    if target_image.shape[0] == 3:
        lossDict['ssim'] = metrics.structural_similarity(target_image, output_image, data_range=255.0, channel_axis=0)
    else:
        lossDict['ssim'] = metrics.structural_similarity(target_image, output_image, data_range=255.0)

    target_image = np.expand_dims(target_image.astype(np.float32), axis=0)
    output_image = np.expand_dims(output_image.astype(np.float32), axis=0)
    # target_image /= 255.0
    # output_image /= 255.0
    target_image = (target_image - 127.5) / 127.5
    output_image = (output_image - 127.5) / 127.5
    # lossDict['fsim'] = fsim(target_image, output_image)
    lossDict['lpips'] = lpips_model(torch.from_numpy(target_image), torch.from_numpy(output_image)).item()

    return lossDict


def main(splitText=None):
    outputDir = r"/home/wrz/src/python/SegRefiner/MyRefiner-VAE/results/25-07-17_13-55-29/depth_npy"

    outputFiles = os.listdir(outputDir)

    labelDir = f"Dataset/BASE_DATA_DIR/ImageToDEM-{targetAreaName}/singleRGBNormalizationTest/DEM_255-unique"  # todo：Test
    # labelFiles = os.listdir(labelDir)
    # assert len(outputFiles) == len(labelFiles), "The number of output files and label files should be the same."
    allMetrics = []
    lpips_model = lpips.LPIPS(net='alex')
    nameList = []
    if splitText:
        targetArray = open(splitText, 'r').read().splitlines()
    for i in tqdm.tqdm(range(len(outputFiles))):
        # if "11081" not in outputFiles[i]:
        #     continue
        index = outputFiles[i].split("_")[1].split(".")[0]
        if splitText and index not in targetArray:
            continue
        target = outputFiles[i].split("tile")[0]
        nameList.append(outputFiles[i])
        # assert f"tile_{index}.tif" in labelFiles
        # assert outputFiles[i].replace("_pred", "") == labelFiles[i].replace("tif", "png")
        if outputFiles[i].endswith(".npy"):
            output = np.load(os.path.join(outputDir, outputFiles[i]))
        else:
            output = np.array(Image.open(os.path.join(outputDir, outputFiles[i])))
        if target:
            label = np.array(Image.open(
                os.path.join(f"Dataset/BASE_DATA_DIR/ImageToDEM-{target}/singleRGBNormalizationTest/DEM_255-unique",
                             f"tile_{index}.tif")))
        else:
            label = np.array(Image.open(os.path.join(labelDir, target + f"tile_{index}.tif")))
        if len(output.shape) > 2:
            assert (output[:, :, 0] == output[:, :, 1]).all() and (output[:, :, 1] == output[:, :, 2]).all()
            output = output[:, :, 0]

        allMetrics.append(calculate_single(label, output, lpips_model))

        for k, v in allMetrics[-1].items():
            if np.isnan(v):
                # raise f"Warning: NaN value found in {k} for image {nameList[i]}"
                print(f"Warning: NaN value found in {k} for image {nameList[i]}")
                allMetrics[-1][k] = 0.0

    avgMetrics = {k: np.mean([m[k] for m in allMetrics]) for k in allMetrics[0].keys()}
    print("Average Metrics:", avgMetrics)


def saveTargetImage(dirPath, targetAreaName, best_images_dict, sourceDir):
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    for area, indices in best_images_dict.items():
        if area not in dirPath:
            continue

        for index in indices:
            image_name = f"tile_{index}_pred_compare.png"
            source_path = os.path.join(sourceDir, image_name)
            if os.path.exists(source_path):
                target_path = os.path.join(dirPath, image_name)
                shutil.copy2(source_path, target_path)
                print(f"saved {image_name} to {dirPath}")
            else:
                print(f"Image {image_name} not found in {sourceDir}, skipping.")


def calculate_process_diff(process_path, label_path):
    step = 6
    diffArray = [[] for _ in range(step)]
    lpips_model = lpips.LPIPS(net='alex')
    ii = 0
    for fname in tqdm.tqdm(os.listdir(label_path), desc="Processing tiles"):
        ii += 1
        index = fname.replace("tile_", "").replace(".tif", "")

        label_file = os.path.join(label_path, f"tile_{index}.tif")
        label = np.array(Image.open(label_file))

        for i in range(step):
            pred_file = os.path.join(process_path, f"tile_{index}_process_depth_{i}.png")
            if not os.path.exists(pred_file):
                print(f"Warning: {pred_file} does not exist, skipping.")
                continue

            output = np.array(Image.open(pred_file))
            if len(output.shape) == 3:
                if (output[:, :, 0] == output[:, :, 1]).all() and (output[:, :, 1] == output[:, :, 2]).all():
                    output = output[:, :, 0]
                else:
                    output = output[:, :, 0]

            try:
                metrics = calculate_single(label, output, lpips_model)
                diffArray[i].append(metrics)
            except Exception as e:
                print(f"Error processing tile {index} at step {i}: {e}")
                continue
        # if ii > 10:
        #     break

    for i in range(step):
        if diffArray[i]:
            avg_metrics = {k: np.mean([m[k] for m in diffArray[i]]) for k in diffArray[i][0].keys()}
            print(f"Step {i} - Average Metrics: {avg_metrics}")
        else:
            print(f"Step {i} - No valid metrics found.")
    # return diffArray


if __name__ == '__main__':
    targetAreaName = "Med"
    splitText = None
    main(splitText=splitText)
