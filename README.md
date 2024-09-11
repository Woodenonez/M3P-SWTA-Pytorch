# Multimodal Motion Prediction based on Sampling WTA+CGF in Pytorch
Use WTA loss with Clustering and Gaussian Fitting (WTA+CGF) to do multimodal motion prediction.

This is a repository presenting multimodal motion prediction with multiple hypothesis estimation with the Clustering and Gaussian Fitting (CGF) method.

#### Publication
The paper is available: [IEEE CASE2022](https://ieeexplore.ieee.org/document/9926544) \
Bibtex citation:
```
@inproceedings{ze_2022_swta,
    author={Zhang, Ze and Dean, Emmanuel and Karayiannidis, Yiannis and Åkesson, Knut},
    booktitle={IEEE 18th International Conference on Automation Science and Engineering (CASE)}, 
    title={Multimodal Motion Prediction Based on Adaptive and Swarm Sampling Loss Functions for Reactive Mobile Robots}, 
    year={2022},
    pages={1110-1115},
    doi={10.1109/CASE49997.2022.9926544}
}
```


#### Requirements
- pytorch
- matplotlib 

#### Data
Two evaluation sets are provided. <br />
Eval 1: [Synthetic] Single-object Interaction Dataset ([SID](https://github.com/Woodenonez/MultimodalMotionPred_SamplingWTACGF_Pytorch/blob/main/src/data_handle/sid_object.py)). <br />
Eval 2: [Realworld] Stanford Drone Dataset ([SDD](https://cvgl.stanford.edu/projects/uav_data/)) ***GO TO BRANCH "sdd_test"***.

#### Model
The model is pre-trained.

#### Test run
All 'main' files are meant to be run. The 'evaluation' file shows the evaluation result. The 'test' file shows a visualized result.
