# DINO-DETR Pedestrian Detection On Custom Dataset
---

Pedestrian Detection on Custom Dataset using an Attention Based Transformer Technique. NB includes visualizations of intermediate layers, sampling points of the best query and fine-tuning apart from regular inference.  

![Annotated Image from the IIT-D Dataset](https://github.com/Aryan-Garg/DINO-DETR_Pedestrian_Detection/blob/95eaab6acafa8bcff2b39179d04392e9d2f27fe9/Screenshot%202023-06-05%20104950.jpg)

---

# DINO 

> This is a standalone notebook of the paper "[DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605)".    


## Checkpoints available here:
We have put our model checkpoints here [[model zoo in Google Drive]](https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_?usp=sharing)[[model zoo in 百度网盘]](https://pan.baidu.com/s/1St5rvfgfPwpnPuf_Oe6DpQ)（提取码"DINO"), where checkpoint{x}_{y}scale.pth denotes the checkpoint of y-scale model trained for x epochs. 

## No installation requirements if using NB on cloud (preferred Google Colab)   

# Bibtex
  
```bibtex
@misc{zhang2022dino,
      title={DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection}, 
      author={Hao Zhang and Feng Li and Shilong Liu and Lei Zhang and Hang Su and Jun Zhu and Lionel M. Ni and Heung-Yeung Shum},
      year={2022},
      eprint={2203.03605},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

@inproceedings{li2022dn,
      title={Dn-detr: Accelerate detr training by introducing query denoising},
      author={Li, Feng and Zhang, Hao and Liu, Shilong and Guo, Jian and Ni, Lionel M and Zhang, Lei},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={13619--13627},
      year={2022}
}

@inproceedings{
      liu2022dabdetr,
      title={{DAB}-{DETR}: Dynamic Anchor Boxes are Better Queries for {DETR}},
      author={Shilong Liu and Feng Li and Hao Zhang and Xiao Yang and Xianbiao Qi and Hang Su and Jun Zhu and Lei Zhang},
      booktitle={International Conference on Learning Representations},
      year={2022},
      url={https://openreview.net/forum?id=oMI9PjOb9Jl}
}
```
