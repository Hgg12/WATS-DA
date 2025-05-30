import Wildlife
import cv2
import numpy as np
from os.path import join, isdir, isfile
from os import mkdir, makedirs
from concurrent import futures
import sys
import time
import re
import json
import os
from tqdm import tqdm
# Print iterations progress (thanks StackOverflow)
def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float32)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_img(id, anns, set_crop_base_path, set_img_base_path, instanc_size=511):
    frame_crop_base_path = join(set_crop_base_path, id)
    if not isdir(frame_crop_base_path): makedirs(frame_crop_base_path)
    # print('{}/{}.jpg'.format(set_img_base_path, id))
    img_path = '{}/{}.jpg'.format(set_img_base_path, id)
    if not os.path.exists(img_path):
        img_path = img_path.replace(".jpg", ".png")
    if not os.path.exists(img_path):
        img_path = img_path.replace(".png", ".JPEG")
    if not os.path.exists(img_path):
        img_path = img_path.replace(".JPEG", ".JPG")
    if not os.path.exists(img_path):
        img_path = img_path.replace(".JPG", ".PNG")
    if not os.path.exists(img_path):
        img_path = img_path.replace(".PNG", ".jpeg")


    im = cv2.imread(img_path)
    avg_chans = np.mean(im, axis=(0, 1))
    # print(anns)
    for trackid, ann in enumerate(anns):
        # XYWHs
        rect = ann
        bbox = [rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]]
        if rect[2] <= 0 or rect[3] <=0:
            continue
        z, x = crop_like_SiamFC(im, bbox, instanc_size=instanc_size, padding=avg_chans)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.z.jpg'.format(0, trackid)), z)
        cv2.imwrite(join(frame_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(0, trackid)), x)


def main(instanc_size=511, num_threads=12):
    dataDir = '/home/w/hgg/Wildlife2024_7_1/'
    crop_path = './train_dataset/Wildlife2024_7_1/crop{:d}'.format(instanc_size)
    if not isdir(crop_path): mkdir(crop_path)

    jsonFile = './train_dataset/Wildlife2024_7_1/result/list.json'
    with open(jsonFile,'r') as file:
        content = file.read()
        pattern = r'\}(\r?\n)\{'
        content = re.sub(pattern,'};\n{',content)

        dictionaries = content.split(';')

    for dictionary in dictionaries:
        data = json.loads(dictionary)
        
        dataType = list(data.keys())[0]
        convert_data = {dataType:data[dataType]}
        set_crop_base_path = join(crop_path, dataType)
        set_img_base_path = join(dataDir, dataType)
        annFile = './train_dataset/Wildlife2024_7_1/result/annotations/{}.json'.format(dataType)
        if isfile(annFile):
            continue
        with open(annFile, 'w') as json_file:
            json.dump(convert_data, json_file)
        coco = Wildlife.Wildlife2024(annFile)
        n_imgs = len(list(coco.imgToAnns.keys()))

        # for id in list(coco.imgToAnns.keys()):
        #     crop_img(id, coco.imgToAnns[id], set_crop_base_path, set_img_base_path, instanc_size)

        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(crop_img, id,
                                  coco.imgToAnns[id],
                                  set_crop_base_path, set_img_base_path, instanc_size) for id in list(coco.imgToAnns.keys())]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                printProgress(i, n_imgs, prefix=dataType, suffix='Done ', barLength=40)
    print('done')


if __name__ == '__main__':
    since = time.time()
    # main(int(sys.argv[1]), int(sys.argv[2]))
    main()
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
