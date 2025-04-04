# covapt_mt
Refactoring of [CovaPT](https://github.com/JayWadekar/CovaPT?tab=readme-ov-file) with a more streamlined user interface and the ability to make (Gaussian) multi-tracer covariance. This repository is primarily geared towards SPHEREx inference developement. 


## Install
You will need a valid version of anaconda / miniconda to use this repository. To install the code, simply download the repository to your local machine and run the following line:

    $ ./install.sh

## Using the Code
Can call it with
```
python -m covapt_mt.scripts.get_covariance ./config/get_covariance.yaml
```

## Credits
This repository builds off of the original code written by Jay Wadekar, Roman Scoccimarro, and Otavio Alves. Any publications or material created that uses this code should cite the original repository (https://arxiv.org/abs/1910.02914).
