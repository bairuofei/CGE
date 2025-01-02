<div align ="center">

<!-- <img src="./assets/logo.png" width="20%"> -->
<h3> IROS 2024: Multi-Robot Active Graph Exploration with Reduced Pose-SLAM Uncertainty via Submodular Optimization </h3>

Ruofei Bai<sup>1,2</sup>, Shenghai Yuan<sup>1</sup>, Hongliang Guo<sup>2</sup>, Pengyu Yin<sup>1</sup>, Wei-Yun Yau<sup>2</sup>, Lihua Xie<sup>1</sup>

<sup>1</sup> Nanyang Technological University,
<sup>2</sup> Institute for Infocomm Research (I2R), Agency for Science, Technology and Research (A*STAR)

<a href="https://ieeexplore.ieee.org/abstract/document/10802691"><img alt="Paper" src="https://img.shields.io/badge/Paper-IEEE%20Xplore-pink"/></a>
<a href="https://arxiv.org/abs/2407.01013"><img alt="Paper" src="https://img.shields.io/badge/Paper-arXiv-8A2BE2"/></a>
<!-- <a href='https://drive.google.com/drive/folders/1UmZ3vA1cOunB-2wgz8T1fJDebhb-gmax?usp=sharing'><img src='https://img.shields.io/badge/Dataset-UMAD-green' alt='Code&Datasets'></a>
<a href="https://www.youtube.com/watch?v=xORb4H-AyNw"><img alt="Video" src="https://img.shields.io/badge/Video-Youtube-red"/></a>
<a href="https://github.com/IMRL/UMAD/blob/main/Doc/UMAD-Poster.pdf"><img alt="Poster" src="https://img.shields.io/badge/Poster-blue"/></a> -->

</div>


# CGE

This repo implements a SLAM-Aware **C**ollaborative **G**raph **E**xploration (CGE) method, which finds quick coverage path for multiple robots, while forming a well-connected collaborative pose graph to reduce SLAM uncertainty.
Approximation algorithms in submodular maximization are adopted to provided performance guarantees for the actively selected loop-closing actions (loop closures).

> This work extends our previous work on single-robot SLAM-aware exploration to the multi-robot case. Follow [this IEEE RA-L paper](https://ieeexplore-ieee-org.remotexs.ntu.edu.sg/document/10577228) and [open-sourced code](https://github.com/bairuofei/Graph-Based_SLAM-Aware_Exploration) for more details.

## News

Our paper has been accpeted by *IEEE/RSJ IROS 2024* !!! 

Please follow [this link](https://arxiv.org/abs/2407.01013) to the Arxiv version. Please consider citing our paper if you find it helpful.
```
@misc{bai2024collaborativegraphexplorationreduced,
      title={Collaborative Graph Exploration with Reduced Pose-SLAM Uncertainty via Submodular Optimization}, 
      author={Ruofei Bai and Shenghai Yuan and Hongliang Guo and Pengyu Yin and Wei-Yun Yau and Lihua Xie},
      year={2024},
      eprint={2407.01013},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2407.01013}, 
}
```


## Requirements

1. Install python libraries `networkx`, `scipy`, `statistics`, `pickle`, `pyyaml`. They can be installed by using `pip install xxx`.

2. Install [OR-Tools](https://developers.google.com/optimization/install) for python: `python -m pip install ortools`.

## Usage

1. Specify save path in `config.yaml`
2. Run `main.py`
3. Visualize the results by running `simulation.py`. The code will read results from paths specified in `config.yaml`.


## Results

Following are the robot's trajectories with (right) & without (left) active loop-closings.

<figure>
    <img src="./image/2robot.gif" alt="Alt Text" width="800" height="400">
    <!-- <figcaption style="text-align:center;">Active TSP-based Method</figcaption> -->
</figure>
</div>


<figure>
    <img src="./image/3robot.gif" alt="Alt Text" width="800" height="400">
    <!-- <figcaption style="text-align:center;">Active TSP-based Method</figcaption> -->
</figure>
</div>


<figure>
    <img src="./image/5robot.gif" alt="Alt Text" width="800" height="400">
    <!-- <figcaption style="text-align:center;">Active TSP-based Method</figcaption> -->
</figure>
</div>


