import os
models=['output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean',\
        'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean2',\
        'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean3']
#models=['output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean2',\
#        'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean3']

for i in models:
    output_dir=os.path.join(i, "generated_masks0516")
    os.system(f"python trial_scripts/generate_crack_mask_conditional_unet_aggexp_embed.py \
              --output_dir {output_dir} --model_dir {i} --sample_count 200  \
              --dtype bfloat16 --seed 0 --post_skeletonize --post_skeletonize_mode erode")