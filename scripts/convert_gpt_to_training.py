import json

if __name__ == "__main__":
    print("hello world!")
    gpt_data = json.load(open('/share/portal/ys749/LLaVA/playground/epic-k-data/EPIC_100_gpt_output_train_v3.json', 'r'))
    for episode in gpt_data:
        for dialogue in episode["conversations"]:
            if "human" == dialogue["from"]:
                dialogue["value"] = "What environment state preceeds this image and what action is being performed?\n<image>"
    json.dump(gpt_data, open('/share/portal/ys749/LLaVA/playground/epic-k-data/EPIC_100_dishwash_llava_train_v3.json', 'w'), indent=4)
