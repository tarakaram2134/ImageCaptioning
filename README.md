# Image Captioning with Xception and LSTM

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Dataset Preparation](#dataset-preparation)
- [Environment Setup](#environment-setup)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)
- [Model Architecture](#model-architecture)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview
This repository implements an **image captioning** pipeline that uses:
1. **Xception** (pre-trained on ImageNet) for extracting 2048-dim image features  
2. **LSTM** network to generate natural-language captions from those features  

It was trained and tested on the Flickr8k dataset.

---

## Directory Structure
```
.
├── descriptions        # cleaned caption text
├── features.p         # serialized image features
├── main.py            # training & evaluation script
├── model.png          # example output visualization
├── test.py            # inference (caption-generation) script
├── tokenizer.p        # serialized tokenizer object
├── README.md          # this document
└── .gitignore         # files/folders to exclude from Git
```

---

## Dataset Preparation
Download the original Flickr8k data from my Google Drive:

- **Images (Flickr8k_Dataset)**  
  https://drive.google.com/drive/folders/1Ww26jM5Ul3_yoa7G53evbxzXXqIe0CgI?usp=sharing  
- **Captions (Flickr8k_text)**  
  https://drive.google.com/drive/folders/1qBI3IHIfa299VsgrZQCV07-JNTIkdRZN?usp=sharing  

1. Download and unzip **Flickr8k_Dataset.zip** into `dataset/Flickr8k_Dataset/`  
2. Download and unzip **Flickr8k_text.zip** into `dataset/Flickr8k_text/`

Your folder tree should then look like:
```
dataset/
├── Flickr8k_Dataset/
└── Flickr8k_text/
```

---

## Environment Setup
```bash
conda create -n image_captioning python=3.11 cudatoolkit=11.8 cudnn=8.6 -c conda-forge
conda activate image_captioning
pip install tensorflow==2.12.0 keras numpy pillow matplotlib tqdm h5py
```

---

## Usage

### Training
```bash
python main.py   --images_dir dataset/Flickr8k_Dataset   --captions_file dataset/Flickr8k_text/Flickr8k.token.txt   --epochs 20 --batch_size 64 --embedding_dim 256 --units 512
```
Checkpoints and preprocessing outputs (`features.p`, `descriptions.txt`, `tokenizer.p`) will be generated in the repo root.

### Inference
```bash
python test.py   --image path/to/your/image.jpg   --tokenizer_file tokenizer.p   --model_checkpoint models2/model_epoch_20.h5
```

---

## Model Architecture
1. **Feature Extractor**: Xception (no top) → 2048-dim vector  
2. **Tokenizer**: fits on `descriptions.txt`, converts words to integer sequences  
3. **Decoder**: Embedding → LSTM → Dense(softmax) over vocabulary  

---

## Dependencies
- Python 3.11  
- TensorFlow 2.12.0  
- Keras  
- NumPy, Pillow, tqdm, Matplotlib, h5py  

---

## Contributing
1. Fork this repo  
2. Create a branch (`git checkout -b feature/my-feature`)  
3. Commit and push your changes  
4. Open a pull request  

---

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
