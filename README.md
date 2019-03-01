<img src='imgs/horse2zebra.gif' align="right" width=384>

<br><br><br>

# Totally unpaired defogging with cycleGAN (CyclePerceptualDefogging)

We provide an extended version of the [original cycleGAN code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) optimized for addressing the problem of defogging.

We added to the original cycleGAN model the concept of [perceptual loss](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf), mainly used in style transfer techniques, and some others improvements specific to the defogging task.

For more information about this project, especially experiments results and comparison with other defogging techniques, please consult my master thesis [here](https://amslaurea.unibo.it/17015) (especially chapter four).

**Note**: The project relies heavily on the original pytorch implementation of cycleGAN, which can be consulted [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). Please, consult the documentation of the original project if you have any doubt or issue about cycleGANs or the project structure.


## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/ggraffieti/pytorch-CycleGAN-and-pix2pix
cd pytorch-CycleGAN-and-pix2pix
```

- Install [PyTorch](http://pytorch.org and) 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, the original project provides the pre-built Docker image and Dockerfile. Please refer to the [Docker](docs/docker.md) page.

### Train the model
- Create your own dataset (we don't provide a default dataset yet)
    - Collect or use an available dataset of unpaired foggy and clear images.
    - Put your foggy images in `dataset/YOUR_DATASET_NAME/trainA` folder.
    - Put your clear images in `dataset/YOUR_DATASET_NAME/trainB` folder.
    - (Optional) Put your foggy test images in `dataset/YOUR_DATASET_NAME/testA`

- Train the model 
```bash
python train.py --dataroot ./dataset/YOUR_DATASET_NAME --name YOUR_MODEL_NAME --model cycle_gan 
```
For other training options, like image resizing, use of multiple GPUs, batch size etc. consult the [tips](docs/tips.md) page.

To see more intermediate results, check out `./checkpoints/YOUR_MODEL_NAME/web/index.html`.


### Test the model
- Yours models are stored in the `./checkpoints` folder. 
- A model can have more than one checkpoint (files `.pth` inside a model folder), a checkout represents the state of a network after the corresponding epoch. 
- For testing a model (generate the defogging images from the foggy test images) run:
```bash
python test.py --dataroot datasets/YOUR_DATASET_NAME/testA --name YOUR_MODEL_NAME --model test
```
The results will be saved at `./results/`.
- For your own experiments, you might want to specify `--netG`, `--norm`, `--no_dropout` to match the generator architecture of the trained model. For all the training options see [tips](docs/tips.md) page.

## Citation
If you use this code for your research, please cite these papers.
```
@thesis{GraffietiThesis,
  title={Style Transfer with Generative Adversarial Networks},
  author={Graffieti, Gabriele},
  type={Master Thesis},
  institution={University of Bologna},
  year={2018}
}

@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networkss},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A},
  booktitle={Computer Vision (ICCV), 2017 IEEE International Conference on},
  year={2017}
}


@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
```

