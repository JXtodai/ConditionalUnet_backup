from pathlib import Path
import os 
imgfolder = Path("/home/jixi/dataset/Diff_img2img/train/input")
text_folder=Path("/home/jixi/dataset/Diff_img2img/train/caption")
os.makedirs(text_folder, exist_ok=True)
png_names = [
    p.name
    for p in imgfolder.iterdir()
    if p.is_file() and p.suffix.lower() == ".png" and not p.name.startswith("._")
]
for i in range(len(png_names)):
    name=png_names[i].split('.')[0]
    agg=""
    exp=""
    if png_names[i][:4]=="L2b1":
        agg="s"
        exp="h"
    elif png_names[i][:4]=="L2b2":
        agg="m"
        exp="h"
    else:
        agg_size=png_names[i][1]
        exp_level=png_names[i][2]
        match agg_size:
            case "1"|"2":
                agg="m"
            case "3"|"4":
                agg="s"
        match exp_level:
            case "0"|"1":
                exp="n"
            case "2"|"3":
                exp="h"
    caption = f"qaAG={agg} qeEXP={exp}"
    txt_path = text_folder/f"{name}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(caption + "\n")