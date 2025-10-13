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
@article{luo2025univid,
  title={UniVid: The Open-Source Unified Video Model},
  author={Luo, Jiabin and Lin, Junhui and Zhang, Zeyu and Wu, Biao and Fang, Meng and Chen, Ling and Tang, Hao},
  journal={arXiv preprint arXiv:2509.24200},
  year={2025}
}
```


## TODO List

- ⬜️ Upload our paper to arXiv and build project pages.
- ⬜️ Upload the code.

## 🏃 Intro UniVid
UniVid is an open-source model that enhances both video generation and video understanding.

Unified video modeling combining generation and understanding capabilities is increasingly important, yet faces two key challenges: maintaining semantic faithfulness during flow-based generation due to text-visual token imbalance and the suboptimality of uniform cross-modal attention across the flow trajectory, and efficiently extending image-centric MLLMs to video without costly retraining. We present UniVid, a unified architecture that couples an MLLM with a diffusion decoder through a lightweight adapter, enabling both video understanding and generation. We introduce Temperature Modality Alignment to improve prompt adherence and Pyramid Reflection for efficient temporal reasoning via dynamic keyframe selection. Extensive experiments on standard benchmarks demonstrate the state-of-the-art performance of our unified video model, achieving a 2.2% improvement on VBench-Long total score compared to the previous SOTA method EasyAnimateV5.1, and 1.0% and 3.3% accuracy gains on MSVD-QA and ActivityNet-QA, respectively, compared with the best prior 7B baselines.

![image](./assets/overall_architecture.jpg)
## 🔧Run Your UniVid

### 1. Install & Requirements

```bash
conda env create -f environment.yaml
conda activate univid
```
### 2. Understanding
Runs the Reflection pipeline on a subset of videos and saves results + traces.
#### Input format：
- video_dir: contains files like video{video_id}.mp4
- gt_file: JSON list with at least video_id, question, answer (optional id)
```json
[{"video_id": 1203, "question": "What color is the car?", "answer": "Red"}]
```
#### Quick Start
```bash
python eval_understanding.py \
  --video_dir /path/to/videos \
  --gt_file /path/to/gt.json \
  --output_dir /path/to/out \
  --output_name subset_run \
  --model_path /path/to/MODEL_DIR \
  --no_ddp_ranker \
  --siglip_ckpt google/siglip2-base-patch16-naflex
```
#### Outputs

- Batch summary: /path/to/out/subset_run.json(fields: id, video_id, question, answer, pred, trace_path)
- Per-sample trace JSONs: /path/to/out/video{video_id}_reflexion.json
- Keyframes (if enabled): sample_frames/video{video_id}/...

#### Note：
If all three rounds fail:

- Static: fallback uses global-caption answer; if insufficient, use the last round.
- Dynamic: fallback uses global-caption answer; if insufficient, use the first round.

For DDP frame ranking, omit --no_ddp_ranker and add --ddp_ranker clip_rank_video_ddp.py --nproc 4.
