import json
import time
import tarfile
import numpy as np
from PIL import Image
import io

import random

# data_path = "/share/portal/ys749/LLaVA/playground/epic-k-data/EPIC_100_dishwash_llava_train_v8-full.json"
data_path = "/share/portal/ys749/LLaVA/playground/epic-k-data/EPIC_100_dishwash_llava_train_v9-full-5rounds-per-chat.json"

start_time = time.time()
list_data_dict = json.load(open(data_path, "r"))
random.shuffle(list_data_dict)
loading_time = time.time() - start_time
print(loading_time)

time_list = []

for e in list_data_dict[:10]
    success = 0
    start_time = time.time()
    try:
        with tarfile.open(e['tar_path']) as tf:
            image_files = [conv['image'] for conv in e['conversations'] if 'image' in conv]
            tarinfos = [tf.getmember(image_file) for image_file in image_files]
            images = [tf.extractfile(tarinfo) for tarinfo in tarinfos]
            images = [image.read() for image in images]
            images = [Image.open(io.BytesIO(image)).convert('RGB') for image in images]

        success = 1

        # image_files = [conv['image'] for conv in self.list_data_dict[i]['conversations'] if 'image' in conv]
        # print(image_files)
        # folder_path, _, _ = self.list_data_dict[i]['tar_path'].partition(".tar")
        # print(folder_path)
        # images = [Image.open(os.path.join(folder_path, image)).convert('RGB') for image in image_files]
    except:
        continue


    end_time = time.time() - start_time
    print(end_time)

    time_list.append(end_time)

print(f"avg: {np.mean(time_list)}, max: {np.max(time_list)}, median: {np.median(time_list)}")
print(f"load_time: {loading_time}")