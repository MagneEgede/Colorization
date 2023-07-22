In this folder the code presented is used for the final Parzen Window Method part in the report:
Comparative Analysis of Colorization Techniques for Grayscale Images.

The method is inspired by the paper:

Charpiat, Guillame et al. : ‘Machine Learning Methods for Automatic Image Colorization’, January 2011
Link: https://www.lri.fr/~gcharpia/colorization_chapter.pdf

The code is written from scratch using snippets of Chat GPT for inspiration, but not for complete functions/methods. It
is written by Magne Egede Rasmussenm, s183963. Having set correct folders and collected pickle files in a correct order,
all scripts should be runable. The order of scripts (and thereby also collecting pickle files) is given below:

Steps for Parzen Window Methods:

The training_code.py will provide descriptors and correct a and b channels values from inputted training images. The
descriptors are based on SIFT and physical interpretation features.

The color_bins.py code takes the a and b channel as input and outputs using Kmeans the most fitting Bins for the
training images. It is the idea that the same bins can be used for the new (unknown) images.

color_bins_visual.py and color_bins_plot.py is simply validation codes for the found color bins.

the test_code.py is the most extensive coding part. It is here the bayesian techniques are performed along the final
preprocessing steps (for example PCA on SIFT output).

I