import json
import numpy as np
import pandas as pd
# data_path = "/share/portal/ys749/LLaVA/playground/epic-k-data/EPIC_100_dishwash_llava_train_v8-full.json"
# data_path = "/share/portal/ys749/LLaVA/playground/epic-k-data/EPIC_100_dishwash_llava_train_v9-full-5rounds-per-chat.json"
data_path = "/share/portal/hw575/LLaVA/playground/epic-k-data/epic-k_val_obj-list_data.json"
# chat_length = 20 # 5 rounds
desc_path = "/share/portal/hw575/LLaVA/playground/epic-k-data/EPIC_descriptions.CSV"

desc = pd.read_csv(desc_path)

list_data_dict = json.load(open(data_path, "r"))
# print(len(list_data_dict))

ep_dict = {}

for ep in list_data_dict:
        ep_name, _, _= ep["ep_id"].partition("__")

        if ep_name not in ep_dict:
                ep_dict[ep_name] = 0
                print(f'{ep_name}: {str(desc.loc[desc["video_id"] == ep_name]["description"].values)}')

        ep_dict[ep_name] += 1

# print("video dictionary: ")
# print(ep_dict)
# print(f"# of episodes: {len(ep_dict)}")
# ep_len = [ep_dict[k] for k in ep_dict]
# print(f"max # of actions: {np.max(ep_len)}")
# print(f"min # of actions: {np.min(ep_len)}")
# print(f"mean # of actions: {np.mean(ep_len)}")
# print(f"total # of actions: {np.sum(ep_len)}")
# new_list_data_dict = []

# for eg in list_data_dict:
#     full_conv = eg['conversations']
#     start_idx = 0
#     counter = 0

#     while start_idx < len(full_conv):
#         new_list_data_dict.append({
#             "id": f"{eg['id']}-{counter}",
#             "conversations": full_conv[start_idx: start_idx + chat_length],
#             "tar_path": eg['tar_path']
#         })

#         start_idx += chat_length
        # counter += 1

# with open("/share/portal/ys749/LLaVA/playground/epic-k-data/EPIC_100_dishwash_llava_train_v11-full-10rounds-per-chat.json", "w") as outf:
#     outf.write(json.dumps(new_list_data_dict))