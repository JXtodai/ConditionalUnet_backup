## Installation
Firstly install diffusers. refer to document on [Huggingface](https://huggingface.co/docs/diffusers/en/installation?install=pip)
```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e ".[torch]"
```
Also install other packages:
```bash
pip install transformers accelerate datasets safetensors
```
Maybe there are other packages need to be checked. Directly put the downloaded diffuser folder in your parent path. for example, your project folder from this github is "C:/Users/jixi/Desktop/genai/", then there should be "diffusers" under this parent folder after git clone the diffusers.

## Folder Tree
### Three files:

datasetprcoessing.ipynb for preparing dataset. Generated dataset is stored under "./data"

train_lora.sh for running training. You can specify the path and parameter inside

main.ipynb for generating images based on trained model. Generated images are stored under "./output"

### Dataset Folders:

Create "images" folder to contain all .tif images.

Create "data" folder to contain .png images, .txt labels, and .json metalabel file. (metalabel file is generated in datasetprocessing.ipynb)

"data" folder used now can be downloaded from the google drive.

Create "val_data" folder to contain .png images and .txt labels for validation (actually this is not used in your training now)

"val_data" now contains images starting from "01150".

## Quick Use Based on trained model:

I have trained two models in folders "model" and "model_text". "model" is for the model trained with aggragate size are numerical description while the "model_txt" for text description )

1. Run accelerate config in bash to interactively configurate the training settings. (details can be asked to ChatGPT. I selected "NO" for most questions)
```bash
accelerate config
```

2. Run main.ipynb to generate images with your text input.

## Usage

1. Run datasetprcoessing.ipynb to generate all required images, labels and metalabel file under folder "images".

2. Run accelerate config in bash to interactively configurate the training settings. (details can be asked to ChatGPT. I selected "NO" for most questions)
```bash
accelerate config
```

3. Run train_lora.sh in bash
```bash
./train_lora.sh
```
trained models are stored under "./model" 

4. Run main.ipynb to generate images and stored in "./output"

## To Do:

Maybe consider how to use val_data and labels inside to quantitatively evaluate the performance. Currently we only rely on huamn eyes to evaluate the performance of generated images.