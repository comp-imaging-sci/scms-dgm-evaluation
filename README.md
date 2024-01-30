## A Method for Evaluating Deep Generative Models of Images via Reproducible High-order Spatial Context<br><sub>Implementation of the post-hoc analyses employed in the paper.</sub>

**A Method for Evaluating Deep Generative Models of Images via Reproducible High-order Spatial Context**<br>
Rucha Deshpande, Mark A. Anastasio, Frank J. Brooks<br>
[Paper](https://arxiv.org/abs/2111.12577v2)<br>
[Data](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HHF4AF)<br> 

Abstract: *Deep generative models (DGMs) have the potential to revolutionize diagnostic imaging. Generative adversarial networks (GANs) are one kind of DGM which are widely employed. The overarching problem with deploying GANs, and other DGMs, in any application that requires domain expertise in order to actually use the generated images is that there generally is not adequate or automatic means of assessing the domain-relevant quality of generated images. In this work, we demonstrate several objective tests of images output by two popular GAN architectures. We designed several stochastic context models (SCMs) of distinct image features that can be recovered after generation by a trained GAN. Several of these features are high-order, algorithmic pixel-arrangement rules which are not readily expressed in covariance matrices. We designed and validated statistical classifiers to detect specific effects of the known arrangement rules. We then tested the rates at which two different GANs correctly reproduced the feature context under a variety of training scenarios, and degrees of feature-class similarity. We found that ensembles of generated images can appear largely accurate visually, and show high accuracy in ensemble measures, while not exhibiting the known spatial arrangements. Furthermore, GANs trained on a spectrum of distinct spatial orders did not respect the given prevalence of those orders in the training data. The main conclusion is that SCMs can be engineered to quantify numerous errors, per image, that may not be captured in ensemble statistics but plausibly can affect subsequent use of the GAN-generated images.*

**Update:** We have recently employed this method to evaluate the behavior of diffusion generative models and are currently exploring other popular DGMs. Some results are reported [here](https://arxiv.org/abs/2309.10817).


## Usage

This method can be employed to: 
- benchmark novel DGMs against existing DGMs,
- prototype novel architectures and loss functions that encode domain knowledge,
- rule out DGMs for a certain task, based on quantifiable error rates. 

## Data

[Datasets](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HHF4AF) for all three SCMs are available on Harvard Dataverse.

In brief, there are three datasets:
1. Flags SCM
- 8 class dataset
- encodes prescribed context via per-image constraints in position, feature-specific intensity distribution, texture (pixel randomness), prevalence

2. Voronoi SCM 
- 4 classes
- encodes prescribed context via per-image constraints in intensity distribution, texture, prevalence

3. Alphabet SCM
- single class 
- encodes prescribed context via per-image constraints in prevalence

## Requirements
A python environment can be setup with Anaconda3 or Miniconda3 using the scm.yml file.
- `conda env create -f scm.yml`
- `conda activate scm` 

Almost all libraries employed in this work are commonly employed libraries for image and statistical analyses. The only exception is [Skan](https://skeleton-analysis.org/), which we employed for the analysis of the Voronoi images.

## Getting started

- Download the [Datasets](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HHF4AF)
- Train a DGM on these data and generate a large image ensemble. We typically generated image ensembles of size 10,000.
- Run the analysis scripts in this repository on the training data to obtain the true baseline.
- Run the same analysis scripts on the generated images.
- Compare the statistics of the training and generated data. This can be done either via the summary printed by the scripts (see metrics below) or via user-designed statistical tests.
- Multiple DGMs can be benchmarked against the training data in this manner.

The analyses scripts can be run as follows:

```.bash
# Run the analysis script on a set of generated images stored in a directory

# Flags SCM
python flags.py --id=0000_dgm1 --images_path=/path/to/image/dataset --save=/output/directory --masks_path=./flags_masks

# Alphabet SCM
python alphabet.py --id=0000_dgm1 --images_path=/path/to/image/dataset --template_path=./letter_templates --save=/output/directory

# Voronoi SCM
python voronoi.py --id=0000_dgm1 --images_path=/path/to/image/dataset --save=/output/directory
```

## Reporting results/ evaluation metrics

Several metrics can be reported or even constructed from these datasets. The following is intended as a starting point for benchmarking.

1. Flags SCM: 
- percentage of images with ambiguous class identity
- percentage of images with incorrect foreground intensity distribution
- percentage of images with incorrect background intensity distribution
- percentage of images with incorrect foreground pixel randomness (Moran's I)
- percentage of images with incorrect background pixel randomness (Moran's I)
- distribution of classes

2. Voronoi SCM:
- percentage of images that are below the acceptable correlation value (default: 0.8, but subjective)
- percentage of images with high grayscale variance per-image (default std.dev.: 5, but subjective)
- distribution of classes
- additional metrics could be derived from the skeleton properties and other mathematical properties of Voronoi diagrams

3. Alphabet SCM:
- percentage of images where all letters in an image are not visually recognizable
- percentage of images that are rejected based on ensemble prevalence
- percentage of images that are rejected based on per-image prevalence of letters
- additional metrics could be derived for paired prevalences, distribution of letters w.r.t position in the image grid.

## Expected filesystem structure

All three scripts require paths to the directory which contains the generated images of type .png. (Data type can be changed but this may require visual checks along the way.)

Scripts for analyzing images from the Alphabet and Flags SCMs require additional inputs: `./letter_templates`, and `./flags_masks` respectively, that are a part of this repository. These paths are to be explicitly provided by the user.

The analysis scripts create the following directories: 
- `full_dataframes` : extracted statistics from each image are written out, 
- `check_image_lists` : lists of image filenames are written out for imperfect images/ perfect images/ images with potential artifacts, which can be visually confirmed, 
- `metrics` : metrics based on the encoded context that can be employed for benchmarking, 
- `plots` (only for the Voronoi SCM) : a plot showing the ensemble distribution of classes.

If several datasets are analyzed, these three folders would be the parent folders and files corresponding to individual datasets would be contained within them, identified by the user-defined id.

## Citation

If you employ this code for your research, please cite out paper:

```
@article{deshpande2023method,
  title={A method for evaluating deep generative models of images via assessing the reproduction of high-order spatial context},
  author={Deshpande, Rucha and Anastasio, Mark A and Brooks, Frank J},
  journal={arXiv preprint arXiv:2111.12577v2},
  year={2023}
}

```


