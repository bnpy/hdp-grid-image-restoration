# hdp-grid-image-restoration
This repository contains pre-trained models and demo code for ICML 2017 paper "From Patches to Images: A Nonparametric Generative Model"

## Prerequisites
#### bnpy
Please first intall the latest version of our bnpy package. Instructions could be found at https://github.com/bnpy/bnpy.

#### Pillow
Pillow is a maintained fork of Python Imaging Library (PIL). It could be installed with **pip** by running
```sh
pip install Pillow
```

## Installation
#### Clone the repository
```sh
git clone https://github.com/bnpy/hdp-grid-image-restoration.git
```

## Running the demo code

```sh
python demo.py
```
It will run the three demos in the `demo.py` file.

The first demo is written in function `demo_eDP()`, in which the Barbara image is first polluted by additive white Gaussian noisy with standard deviation 25, and then gets denoised by our external DP Grid method. The denoised image would be saved in png format, and should match the middle right plot shown in Figure 3 of our paper.

The second demo is written in function `demo_HDP()`, in which the airplane image is polluted by the same amount of noise as above. It would get denoised by our HDP Grid method. The saved output should match the bottom right plot in Figure 8.

The last demo is written in function `demo_inpainting()`, where the HDP Grid method would inpaint the New Orleans image, and the saved output should match the bottom right plot shown in Figure 7.

Depending on the speed of your computer, the two denoising demos may each take up to 15~30 minutes to run, and the inpainting one could take about two hours.


## Reference

    @inproceedings{ji2017patches,
        title={From Patches to Images: A Nonparametric Generative Model},
        author={Ji, Geng and Hughes, Michael C and Sudderth, Erik B},
        title={International Conference on Machine Learning},
        year={2017}
    }

For questions, please contact: Geng Ji (gji1@uci.edu).
