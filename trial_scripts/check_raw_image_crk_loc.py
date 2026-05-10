
from pathlib import Path
from PIL import Image
import os
import numpy as np


def mask_crk(crk_img, raw_img, agg_img=None, agg_alpha=0.4):
    crk_img = np.asarray(crk_img.convert("L"))
    raw_img = np.asarray(raw_img.convert("RGB")).copy()

    raw_img[:,:,0]=np.where(crk_img!=0,255,raw_img[:,:,0])
    for i in range(1,3):
        raw_img[:,:,i]=np.where(crk_img!=0,0,raw_img[:,:,i])

    if agg_img is None:
        return Image.fromarray(raw_img, mode="RGB")

    agg_img = agg_img.convert("L")
    if agg_img.size != (raw_img.shape[1], raw_img.shape[0]):
        agg_img = agg_img.resize((raw_img.shape[1], raw_img.shape[0]), resample=Image.NEAREST)

    agg_mask = np.asarray(agg_img) != 0
    agg_color = np.array([0, 180, 255], dtype=np.float32)
    blended = raw_img.astype(np.float32)
    blended[agg_mask] = (1.0 - agg_alpha) * blended[agg_mask] + agg_alpha * agg_color

    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8), mode="RGB")

crk_folder="/home/jixi/dataset/Train_conditionalUnet/crk_mask"
agg_folder="/home/jixi/dataset/Train_conditionalUnet/agg_crk_unetpred"
raw_image_folder="/home/jixi/dataset/Train_conditionalUnet/input"
masked_crack_folder="/home/jixi/dataset/Train_conditionalUnet/masked_cracks_check"
os.makedirs(masked_crack_folder, exist_ok=True)
files=[f for f in Path(raw_image_folder).glob("*.png") if not f.stem.startswith("._")]
for f in files:
    crk_name = f"{crk_folder}/{f.stem}.png"
    agg_name = f"{agg_folder}/{f.stem}.png"
    crack=Image.open(crk_name)
    aggregate=Image.open(agg_name)
    print(crack.size)
    raw=Image.open(f)
    masked=mask_crk(crack, raw, aggregate)
    masked.save(f"{masked_crack_folder}/{f.stem}_masked.png")
