import json

if __name__ == "__main__":
    gpt_data = json.load(open('/share/portal/ys749/LLaVA/playground/epic-k-data/EPIC_100_image_val.json', 'r'))["prompts"]
    for episode in gpt_data:
        for dialogue in episode["conversations"]:
            if "human" == dialogue["from"]:
                dialogue["value"] = "What environment state preceeds this image and what action is being performed?\n<image>"
    json.dump(gpt_data, open('/share/portal/ys749/LLaVA/playground/epic-k-data/EPIC_100_dishwash_llava_val.json', 'w'), indent=4)
