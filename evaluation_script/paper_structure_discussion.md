please help me to clarify the structure of a manuscript to be written on the conditionl diffusion model.
I trained conditional diffusion model based on the original pipeline "diffusers/examples/unconditional_image_generation/train_unconditional.py"
The modifications include: added class label combo to distinguish bewteen different physics constraints, added losses including BCE, DICE, area matching loss, overlap matching loss and clDICE
The training1 reached a relatively idea results after n epochs, when i set diffusion, BCE, DICE loss=0.5, respectively, area matching, overlap matching loss= 5, clDICE loss=0. Then I trained based on that with setting different values for each loss. 
My question is 1. whether I need to train a more basic model as baseline using the original uncontitional model. 2. How to structure the paper with the above training results. 
Please provide suggestions without modifying any script.