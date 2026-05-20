import os
#'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean',\
models=['output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean_refer',\
         'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline1',\
         'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline2']
for i in models:
    gen_dir=os.path.join(i, "generated_masks") 
    os.system(f"python evaluation_script/evaluate_generated_masks3.py --gen_dir {gen_dir}")
