import os
models=['output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean',\
        'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean2',\
        'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean3']
for i in models:
    gen_dir=os.path.join(i, "generated_masks0516") 
    os.system(f"python evaluation_script/evaluate_generated_masks.py --gen_dir {gen_dir}")
