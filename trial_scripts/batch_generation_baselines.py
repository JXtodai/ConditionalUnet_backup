import os 
models=['output_conditional_unet_aggexp_embed_full_x0pred_cldice6_clean_refer',\
        'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline1',\
         'output_conditional_unet_aggexp_embed_full_x0pred_cldice6_baseline2']   
generation_scripts=['trial_scripts/generate_crack_mask_conditional_unet_aggexp_embed.py',\
                    'trial_scripts/generate_crack_mask_conditional_unet_aggexp_embed_baseline1.py',\
                    'trial_scripts/generate_crack_mask_conditional_unet_aggexp_embed_baseline2.py']
for i in range(3):
    os.system(f"python {generation_scripts[i]} \
                  --output_dir {models[i]}/generated_masks --model_dir {models[i]} --sample_count 200  \
                  --dtype bfloat16 --seed 0 --post_skeletonize --post_skeletonize_mode erode")