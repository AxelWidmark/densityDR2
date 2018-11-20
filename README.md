# densityDR2

This project contains code used to produce the results of "Measuring the local matter density using Gaia DR2" (Widmark, 2018, arXiv:XXXXXXXXX).

This code is written in Python and uses the following packages:
numpy, scipy, healpy, matplotlib, pandas

The code consists of four scripts:

* HierarchicalModel.py -- This is the main script, and contains the class 'stellarsample'. When initialised, it loads the specified data sample. The function 'lnposterior' can then be called, which takes a vector of population parameters as input, and returns the natural log of posterior value (plus some constant). This function can be sampled with an MCMC. All the necessary data to run this script for samples S1-S8, and M1-M4, is present in the ./Data catalogue. For more details, see comments in the code.
* CalculateEffectiveArea.py -- This code calculated the effective area of some sample (which is not necessary to run, as the effective areas of all samples are already computed and found in the ./Data directory). The code is very slow, and if you wish to run it, I recommend parallelising it first.
* MakeDataSample.py -- Contains two functions that can be called to (1) compile a data sample from the Gaia catalogue according to the sample cuts, or (2) generate a mock data sample. In the former case, the Gaia catalogue must be supplied (it is not included in this repository).
* ConvertCoordDR2.py -- A script for some coordinate transformations, necessary for running MakeDataSample.py.

The readme and the code might be updated and made more user friendly. If you have any questions, please email me (axel.widmark@fysik.su.se).

If you want to use parts of this code, please acknowledge this work by citing the article.
