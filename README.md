# <img src="https://github.com/AIGeeksGroup/UniVid/blob/website/assets/univid_logo.png" alt="logo" width="25"/> UniVid: The Open-Source Unified Video Model
This is the official repository for the paper:
> **UniVid: The Open-Source Unified Video Model**
>
> [Jiabin Luo](https://king-play.github.io/)\*, [Junhui Lin](https://github.com/kmp1001)\*, [Zeyu Zhang](https://steve-zeyu-zhang.github.io/)\*<sup>†</sup>, Biao Wu\*, Meng Fang, Ling Chen, and [Hao Tang](https://ha0tang.github.io/)<sup>‡</sup>  
>
> *Equal contribution. <sup>†</sup>Project lead. <sup>‡</sup>Corresponding author.
>
> ### [Paper](https://arxiv.org/abs/2509.24200) | [Website](https://aigeeksgroup.github.io/UniVid) | [Models](https://huggingface.co/AIGeeksGroup/UniVid) | [HF Paper](https://huggingface.co/papers/2509.24200)


https://github.com/user-attachments/assets/e81d2e6e-7a69-48a0-91aa-f0a38b5498d2





## ✏️ Citation

If you find our code or paper helpful, please consider starring ⭐ us and citing:

```
@article{luo2025unividopensourceunifiedvideo,
      title={UniVid: The Open-Source Unified Video Model}, 
      author={Jiabin Luo and Junhui Lin and Zeyu Zhang and Biao Wu and Meng Fang and Ling Chen and Hao Tang},
      journal={https://arxiv.org/abs/2509.24200}, 
      year={2025}
}
```


## TODO List

- ⬜️ Upload our paper to arXiv and build project pages.
- ⬜️ Upload the code.

## 🏃 Intro UniVid
UniVid combines video understanding and generation using an MLLM with a diffusion decoder, achieving state-of-the-art performance through Temperature Modality Alignment and Pyramid Reflection.

Unified video modeling combining generation and understanding capabilities is increasingly important, yet faces two key challenges: maintaining semantic faithfulness during flow-based generation due to text-visual token imbalance and the suboptimality of uniform cross-modal attention across the flow trajectory, and efficiently extending image-centric MLLMs to video without costly retraining. We present UniVid, a unified architecture that couples an MLLM with a diffusion decoder through a lightweight adapter, enabling both video understanding and generation. We introduce Temperature Modality Alignment to improve prompt adherence and Pyramid Reflection for efficient temporal reasoning via dynamic keyframe selection. Extensive experiments on standard benchmarks demonstrate the state-of-the-art performance of our unified video model, achieving a 2.2% improvement on VBench-Long total score compared to the previous SOTA method EasyAnimateV5.1, and 1.0% and 3.3% accuracy gains on MSVD-QA and ActivityNet-QA, respectively, compared with the best prior 7B baselines.

![image](./assets/overall_architecture.jpg)
## 🔧Run Your UniVid

### 1. Install & Requirements

```bash
conda env create -f environment.yaml
conda activate univid
```
