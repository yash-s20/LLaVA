import json
import os
import tarfile
import io
import code

from llava.api.llava_query import LLaVAQuery

from PIL import Image


if __name__ == "__main__":
    sys_msg = ("You are a helpful language and vision assistant. Each time, the user will provide a first-person, "
               "egocentric image and a list of objects that are highly likely to appear in the image. "
               "Not all objects might show up in the image because they could be occluded. "
               "You should not describe any objects that are not listed. In your responses, you must answer "
               "the question that the human has specific to this scene and objects listed")
    question_template = "<image>\nThe list of objects that might be in this image is: <obj_list>. <question>"
    ll_mod = LLaVAQuery('checkpoints/llava-llama-2-7b-chat-hf-lightning-merge')
    with tarfile.open('/share/portal/ys749/EPIC-KITCHENS/P18/rgb_frames_val/P18_02.tar') as tf:
        # print(tf)
        tarinfo = tf.getmember('./frame_0000010294.jpg')
        image = tf.extractfile(tarinfo)
        image = image.read()
        image = Image.open(io.BytesIO(image)).convert('RGB')
    OBJECTS = input("Objects: ")
    q = input("First question: ")
    
    ans, conv = ll_mod.query_one(image, sys_msg, question_template.replace("<obj_list>", OBJECTS).replace("<question>", q), True)
    print(ans)
    try:
        while True:
            q = input("Next question: ")
            conv.append({
                "from": "USER",
                "value": q
            })
            ans, conv = ll_mod.query_conv(image, sys_msg, conv, True)
            print("llava: " + ans)
    except KeyboardInterrupt as e:
        print("Taking over control")
        code.interact()
