#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse
import subprocess
import time
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModel
from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

from openai import OpenAI


def parse_args():
    p = argparse.ArgumentParser("Reflexion-BAGEL (batch subset eval) — same pipeline, new data loader")
    p.add_argument("--video_dir", required=True, help="Directory containing videos named like video{video_id}.mp4")
    p.add_argument("--gt_file", required=True, help="JSON with entries containing video_id, question, answer")
    p.add_argument("--output_dir", required=True, help="Directory to save batch summary and per-sample traces")
    p.add_argument("--output_name", required=True, help="Batch summary filename without .json")
    p.add_argument("--id_from", type=int, required=True, help="Inclusive lower bound of video_id")
    p.add_argument("--id_to", type=int, required=True, help="Inclusive upper bound of video_id")

    p.add_argument("--bagel_model_path", required=True,
                   help="BAGEL model dir with llm_config.json / vit_config.json / ema.safetensors / ae.safetensors")
    p.add_argument("--siglip_ckpt", default="google/siglip2-base-patch16-naflex")
    p.add_argument("--device", default="cuda:0", help="Device for SigLIP2 when using single-GPU ranking")
    p.add_argument("--no_ddp_ranker", action="store_true", help="Use built-in single-GPU SigLIP2 ranking")
    p.add_argument("--ddp_ranker", default="clip_rank_video_ddp.py", help="DDP ranking script path")
    p.add_argument("--nproc", type=int, default=4, help="Processes for torchrun --nproc_per_node")

    p.add_argument("--static_seq", default="4,8,16", help="Static pyramid sequence")
    p.add_argument("--dynamic_seq", default="64,32,16", help="Dynamic pyramid sequence (with MMR)")

    p.add_argument("--pool_frames", type=int, default=64, help="Uniformly sampled candidate pool size")
    p.add_argument("--siglip_bs", type=int, default=64, help="SigLIP2 batch size")
    p.add_argument("--save_frames_root", default="sample_frames", help="Directory to save selected keyframes")

    p.add_argument("--deepseek_api_key", default=os.getenv("DEEPSEEK_API_KEY", ""))

    p.add_argument("--max_think_token_n", type=int, default=512)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--temperature", type=float, default=0.3)

    p.add_argument("--max_mem_per_gpu", default="80GiB", help="Per-GPU max memory for device_map inference")
    p.add_argument("--offload_folder", default="/tmp/offload", help="CPU/offload folder")

    p.add_argument("--video_exts", nargs="*", default=[".mp4", ".avi", ".mov", ".mkv"], help="Video file extensions")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--print_plan", action="store_true", help="(No-op) kept for compatibility")
    return p.parse_args()


def set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _sample_indices(n_total: int, num: int):
    num = max(1, min(num, n_total))
    if n_total <= 1:
        return [0]
    return np.linspace(0, n_total - 1, num=num, dtype=int).tolist()


def _read_video_decord(path: str, num_frames: int):
    from decord import VideoReader, cpu
    vr = VideoReader(path, ctx=cpu(0))
    n = len(vr)
    idx = _sample_indices(n, num_frames)
    frames = []
    for i in idx:
        arr = vr[i].asnumpy()
        frames.append(Image.fromarray(arr))
    return frames


def _read_video_torchvision(path: str, num_frames: int):
    from torchvision.io import read_video
    vframes, _, _ = read_video(path, pts_unit="sec")
    n = int(vframes.shape[0])
    if n == 0:
        raise RuntimeError("torchvision.read_video got 0 frames")
    idx = _sample_indices(n, num_frames)
    frames = []
    for i in idx:
        arr = vframes[i].numpy()
        if arr.dtype != 'uint8':
            arr = arr.astype('uint8')
        frames.append(Image.fromarray(arr))
    return frames


def _read_video_opencv(path: str, num_frames: int):
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("opencv failed to open video")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if n <= 0:
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("opencv reopen failed")
    idx = set(_sample_indices(n, num_frames))
    frames = []
    t = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if t in idx:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(fr))
        t += 1
    cap.release()
    if not frames:
        raise RuntimeError("opencv read 0 frames")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return frames


def sample_video_frames_uniform(video_path: str, num_frames: int = 64) -> List[Image.Image]:
    last_err = None
    try:
        import decord  # noqa: F401
        return _read_video_decord(video_path, num_frames)
    except Exception as e:
        last_err = e
    try:
        return _read_video_torchvision(video_path, num_frames)
    except Exception as e:
        last_err = e
    try:
        return _read_video_opencv(video_path, num_frames)
    except Exception as e:
        last_err = e
    raise RuntimeError(f"Failed to decode video. Last error: {last_err}")


class Siglip2Scorer:
    def __init__(self, ckpt: str, device: str = "cuda:0", dtype: torch.dtype = torch.float16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.proc = AutoProcessor.from_pretrained(ckpt)
        self.model = AutoModel.from_pretrained(ckpt).to(self.device, dtype=self.dtype).eval()

    @torch.no_grad()
    def emb_text(self, q: str) -> torch.Tensor:
        t_inputs = self.proc(text=[q], return_tensors="pt").to(self.device)
        with torch.autocast("cuda", dtype=self.dtype, enabled=(self.device.type == "cuda")):
            t = self.model.get_text_features(**t_inputs)
        return torch.nn.functional.normalize(t, dim=-1)

    @torch.no_grad()
    def emb_imgs(self, images: List[Image.Image], bs: int = 64) -> torch.Tensor:
        vecs = []
        for i in range(0, len(images), bs):
            batch = images[i:i + bs]
            x = self.proc(images=batch, return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=self.dtype, enabled=(self.device.type == "cuda")):
                v = self.model.get_image_features(**x)
            v = torch.nn.functional.normalize(v, dim=-1)
            vecs.append(v)
        return torch.cat(vecs, dim=0) if vecs else torch.empty(0, 1024, device=self.device)

    @torch.no_grad()
    def rank_frames(self, frames: List[Image.Image], query: str, topk: int, bs: int = 64) -> Tuple[List[int], List[float]]:
        if len(frames) == 0:
            return [], []
        t = self.emb_text(query)
        v = self.emb_imgs(frames, bs=bs)
        sims = (v @ t.T).squeeze(-1).float()
        k = min(topk, sims.shape[0])
        vals, idx = torch.topk(sims, k=k)
        return idx.tolist(), [float(v) for v in vals.tolist()]


def ddp_select_topk_frames(video_path: str, query: str, topk: int, num_frames: int, ckpt: str, nproc: int) -> Dict[str, Any]:
    cmd = [
        "torchrun", f"--nproc_per_node={nproc}",
        "clip_rank_video_ddp.py",
        "--video", video_path,
        "--query", query,
        "--ckpt", ckpt,
        "--topk", str(topk),
        "--num_frames", str(num_frames),
        "--out_dir", "sample_frames"
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", "ignore")
    last_line = out.strip().splitlines()[-1]
    return json.loads(last_line)


def mmr_select(embs: torch.Tensor, query_emb: torch.Tensor, K: int, lam: float = 0.5) -> List[int]:
    sims_q = (embs @ query_emb.T).squeeze(-1)
    N = embs.shape[0]
    selected = []
    candidate = set(range(N))
    sims_ii = embs @ embs.T
    while len(selected) < min(K, N) and len(candidate) > 0:
        best_i, best_score = None, -1e9
        for i in candidate:
            div = 0.0 if not selected else torch.max(sims_ii[i, selected]).item()
            score = lam * sims_q[i].item() - (1.0 - lam) * div
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)
        candidate.remove(best_i)
    return selected


class Qwen:
    def __init__(self, api_key: str):
        assert api_key, "Qwen API key is required (--deepseek_api_key or env DEEPSEEK_API_KEY)."
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = "qwen-plus"

    def chat(self, sys_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_prompt}],
            stream=False
        )
        return resp.choices[0].message.content

    def eval_answer(self, question: str, global_caption: str, answer: str) -> Dict[str, Any]:
        import json as _json, re
        sys_p = (
            "You are a precise evaluator for video-QA. "
            "Return a SINGLE-LINE JSON ONLY. No Markdown, no code block, no extra text. "
            "Keys: score (float 0..1), verdict ('accept' if score>=0.7 else 'reject'), brief_reason (string; 1-2 short bullets)."
        )
        one_shot_user = (
            "Question: How many times does the dog appear?\n"
            "Global Caption: A brown dog runs into the yard; later the same dog returns with a ball.\n"
            "Candidate Answer: The dog appears twice."
        )
        one_shot_assistant = '{"score": 0.92, "verdict": "accept", "brief_reason": "Counts match frames; consistent with caption."}'
        real_user = f"""Question: {question}
Global Caption: {global_caption}
Candidate Answer: {answer}

Output strictly one-line JSON as in the example. Do not explain.
"""
        user_p = "[EXAMPLE]\n" + one_shot_user + "\n\n[EXAMPLE ASSISTANT]\n" + one_shot_assistant + "\n\n[YOUR TASK]\n" + real_user
        txt = self.chat(sys_p, user_p).strip()

        def _parse_json(s: str) -> Dict[str, Any]:
            try:
                return _json.loads(s)
            except Exception:
                m = re.search(r'(\{.*\}|\[.*\])', s, flags=re.S)
                if m:
                    try:
                        return _json.loads(m.group(1))
                    except Exception:
                        pass
            return {}
        obj = _parse_json(txt)

        def _coerce(d, k, default):
            return d[k] if isinstance(d, dict) and k in d else default
        score = _coerce(obj, "score", 0.0)
        try:
            score = float(score); score = max(0.0, min(1.0, score))
        except Exception:
            score = 0.0
        verdict = _coerce(obj, "verdict", "accept" if score >= 0.7 else "reject")
        verdict = "accept" if str(verdict).lower().strip() == "accept" and score >= 0.7 else "reject"
        brief = str(_coerce(obj, "brief_reason", "")).strip() or "Insufficient evidence or mismatch."
        return {"score": score, "verdict": verdict, "brief_reason": brief}

    def summarize_frames(self, frame_captions: List[str]) -> str:
        sys_p = (
            "You are a precise video-summary assistant. "
            "Summarize chronologically ordered frame notes into a compact global caption. "
            "Do not invent facts; only use what appears in the notes."
        )
        user_p = (
            "Frame-wise notes (chronological, earlier→later):\n"
            f"{chr(10).join(f'- {c}' for c in frame_captions[:64])}\n\n"
            "Write ONE global caption that connects multiple frames focusing on visual facts only."
        )
        return self.chat(sys_p, user_p).strip()

    def classify_qtype(self, question: str) -> dict:
        sys_p = "You are a precise QA type classifier for video questions. Output JSON only."
        user_p = f"""
Decide whether the following video question requires temporal reasoning ("dynamic")
or can be answered from a small set of frames without ordering ("static").

- "dynamic": needs counting/repetition/order/temporal dependency.
- "static": identity/attribute/location/one-shot action.

Question:
{question}

Return a JSON with fields:
- qtype: "static" or "dynamic"
- rationale: 1-2 short phrases
"""
        txt = self.chat(sys_p, user_p).strip()
        try:
            obj = json.loads(txt)
            qtype = str(obj.get("qtype", "static")).lower().strip()
            if qtype not in ("static", "dynamic"):
                qtype = "static"
            return {"qtype": qtype, "rationale": obj.get("rationale", "")}
        except Exception:
            return {"qtype": "static", "rationale": "fallback"}

    def answer_from_global(self, question: str, global_caption: str) -> str:
        sys_p = "You answer concisely using only the given question and the global video caption."
        user_p = f"""Question: {question}
Global caption (may miss fine details): {global_caption}

Instruction:
- Produce a single short answer (1-2 sentences).
- If information is insufficient, say 'Not enough evidence from global caption.'"""
        return self.chat(sys_p, user_p).strip()


class DeepSeek:
    def __init__(self, api_key: str):
        assert api_key, "DeepSeek API key is required (--deepseek_api_key or env DEEPSEEK_API_KEY)."
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.model = "deepseek-v3.1"

    def chat(self, sys_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_prompt}],
            stream=False
        )
        return resp.choices[0].message.content

    def reflect(self, question: str, global_caption: str, last_answer: str, eval_json: Dict[str, Any]) -> Dict[str, str]:
        import json as _json, re
        sys_p = (
            "You are the Reflector in a video-understanding pipeline. "
            "Output JSON ONLY with a single key: refined_query (<=25 tokens, declarative)."
        )
        one_shot_user = f"""
[Example]
Question: "What sport is the athlete in a green jersey playing?"
Global Caption: "Multiple athletes are running on a field; one person wears a green jersey; later people gather near the sideline."
Last Answer: "He is playing basketball."
Evaluation JSON: {{"score": 0.32, "verdict": "reject", "brief_reason": "wrong activity"}}
Return:
{{"refined_query": "Green-jersey athlete kicks a ball with his foot"}}

Now CURRENT CASE:
Question: {question}
Global Caption: {global_caption}
Last Answer: {last_answer}
Evaluation JSON: {_json.dumps(eval_json, ensure_ascii=False)}
"""
        txt = self.chat(sys_p, one_shot_user).strip()
        def _extract_json_blob(s: str) -> str:
            m = re.search(r"\{.*\}", s, flags=re.DOTALL)
            return m.group(0) if m else s
        try:
            obj = _json.loads(_extract_json_blob(txt))
            rq = str(obj.get("refined_query", "")).strip()
        except Exception:
            rq = ""
        return {"refined_query": rq}


def _make_reflection_clients(api_key: str):
    if api_key:
        return DeepSeek(api_key=api_key), Qwen(api_key=api_key)

    class _NoOpReflector:
        def reflect(self, *args, **kwargs):
            return {"refined_query": ""}

    class _NoOpQwen:
        def classify_qtype(self, question: str):
            return {"qtype": "static", "rationale": "no-api-key"}
        def summarize_frames(self, frame_captions):
            return ""
        def eval_answer(self, question, global_caption, answer):
            return {"score": 0.0, "verdict": "reject", "brief_reason": "no-api-key"}
        def answer_from_global(self, question, global_caption):
            return "Not enough evidence from global caption."

    return _NoOpReflector(), _NoOpQwen()


def load_bagel_and_inferencer(model_path: str, max_mem_per_gpu: str, offload_folder: str) -> InterleaveInferencer:
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    bagel_cfg = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, bagel_cfg)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    max_memory = {i: max_mem_per_gpu for i in range(torch.cuda.device_count())}
    device_map = infer_auto_device_map(
        model,
        max_memory=max_memory,
        no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
    )
    same_device_modules = [
        "language_model.model.embed_tokens",
        "time_embedder",
        "latent_pos_embed",
        "vae2llm",
        "llm2vae",
        "connector",
        "vit_pos_embed",
    ]
    if torch.cuda.device_count() == 1:
        first_device = device_map.get(same_device_modules[0], "cuda:0")
        for k in same_device_modules:
            device_map[k] = first_device
    else:
        first_device = device_map.get(same_device_modules[0], list(device_map.values())[0])
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device

    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map=device_map,
        offload_buffers=True,
        dtype=torch.bfloat16,
        force_hooks=True,
        offload_folder=offload_folder,
    ).eval()

    inferencer = InterleaveInferencer(
        model=model,
        vae_model=vae_model,
        tokenizer=tokenizer,
        vae_transform=vae_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    return inferencer


def save_topk_frames(save_root: str, video_path_or_name: str, stage_tag: str,
                     frames: List[Image.Image], global_indices: List[int], scores: Optional[List[float]] = None) -> str:
    os.makedirs(save_root, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path_or_name))[0]
    out_dir = os.path.join(save_root, video_name, stage_tag)
    os.makedirs(out_dir, exist_ok=True)
    for i, gi in enumerate(global_indices):
        if i >= len(frames):
            break
        sc = (scores[i] if scores and i < len(scores) else None)
        fname = f"{i:03d}_idx{gi}.jpg" if sc is None else f"{i:03d}_idx{gi}_score{sc:.4f}.jpg"
        frames[i].convert("RGB").save(os.path.join(out_dir, fname), format="JPEG", quality=95)
    return out_dir


def reflexion_answer_one(
    video_path: str,
    question: str,
    bagel: InterleaveInferencer,
    ds_client,
    qwen_client,
    args
) -> Tuple[str, Dict[str, Any]]:
    _ = time.time()

    qtype_info = qwen_client.classify_qtype(question)
    qtype = qtype_info.get("qtype", "static")

    pool_frames = sample_video_frames_uniform(video_path, num_frames=args.pool_frames)
    N = len(pool_frames)

    cap_seed_idx = _sample_indices(N, 16)
    cap_seed_frames = [pool_frames[i] for i in cap_seed_idx]
    frame_notes = []
    SINGLE_FRAME_PROMPT = (
        "You are assisting video understanding via per-frame analysis. "
        "Describe the main objects and actions in THIS SINGLE FRAME concisely."
    )
    for fr in cap_seed_frames:
        out = bagel(
            image=fr,
            text=SINGLE_FRAME_PROMPT,
            understanding_output=True,
            max_think_token_n=args.max_think_token_n,
            do_sample=args.do_sample,
            text_temperature=args.temperature
        )
        frame_notes.append(out.get("text", ""))
    global_caption = qwen_client.summarize_frames(frame_notes)

    scorer = None
    if args.no_ddp_ranker:
        scorer = Siglip2Scorer(args.siglip_ckpt, device=args.device, dtype=torch.float16)

    def select_topk_from_pool(query_text: str, topk: int, exclude: set) -> Tuple[List[int], List[float]]:
        remain_idx = [i for i in range(N) if i not in exclude]
        remain_frames = [pool_frames[i] for i in remain_idx]
        if len(remain_frames) == 0:
            return [], []
        if args.no_ddp_ranker:
            idx_local, sc = scorer.rank_frames(remain_frames, query_text, topk=min(topk, len(remain_frames)),
                                               bs=args.siglip_bs)
            chosen = [remain_idx[j] for j in idx_local]
            return chosen, sc
        else:
            info = ddp_select_topk_frames(video_path, query_text, topk=min(topk, len(remain_frames)),
                                          num_frames=args.pool_frames, ckpt=args.siglip_ckpt, nproc=args.nproc)
            return info["frame_indices"], info["scores"]

    def bagel_qa_on_frames(frames: List[Image.Image], question: str) -> str:
        out = bagel.video_understanding(
            video=frames,
            text=question,
            fps=1.0,
            max_frames=len(frames),
            max_pixels=2000*2000,
            think=False,
            max_think_token_n=args.max_think_token_n,
            do_sample=args.do_sample,
            text_temperature=args.temperature,
        )
        return out.get("text", "")

    trace: Dict[str, Any] = {
        "video": video_path,
        "question": question,
        "qtype_init": qtype,
        "global_caption": global_caption,
        "rounds": []
    }
    refined_query = question

    if qtype == "static":
        seq = [int(x) for x in args.static_seq.split(",")]
        selected: List[int] = []
        exclude = set()
        last_bagel_answer = ""
        for it, K in enumerate(seq, start=1):
            need = K - len(selected)
            if need > 0:
                new_idx, _ = select_topk_from_pool(refined_query, need, exclude)
                selected.extend(new_idx)
                for x in new_idx:
                    exclude.add(x)

            stage_tag = f"static_it{it}_k{len(selected)}"
            frames_this = [pool_frames[i] for i in selected]
            save_topk_frames(args.save_frames_root, video_path, stage_tag, frames_this, selected, None)

            ans = bagel_qa_on_frames(frames_this, question)
            last_bagel_answer = ans
            eval_json = qwen_client.eval_answer(question, global_caption, ans)

            trace["rounds"].append({
                "type": "static",
                "iter": it,
                "K": len(frames_this),
                "answer": ans,
                "eval": eval_json
            })

            ok = (eval_json.get("verdict", "reject") == "accept")
            try:
                ok = ok or (float(eval_json.get("score", 0)) >= 0.7)
            except Exception:
                pass
            if ok:
                final_answer = ans
                break

            refl = ds_client.reflect(question, global_caption, ans, eval_json)
            refined_query = refl.get("refined_query", refined_query)
        else:
            fallback = qwen_client.answer_from_global(question, global_caption).strip()
            if fallback == "" or ("not enough" in fallback.lower() or "insufficient" in fallback.lower()):
                final_answer = last_bagel_answer
                trace["fallback"] = {"reason": "final_score_below_0.7_and_global_not_enough",
                                     "answer_from_qwen": fallback}
            else:
                final_answer = fallback
                trace["fallback"] = {"reason": "final_score_below_0.7", "answer_from_qwen": fallback}

    else:
        seq = [int(x) for x in args.dynamic_seq.split(",")]
        K0 = seq[0]
        idx0 = _sample_indices(N, K0)
        frames0 = [pool_frames[i] for i in idx0]
        save_topk_frames(args.save_frames_root, video_path, f"dynamic_it1_k{K0}", frames0, idx0, None)
        ans0 = bagel_qa_on_frames(frames0, question)
        eval0 = qwen_client.eval_answer(question, global_caption, ans0)
        trace["rounds"].append({
            "type": "dynamic", "iter": 1, "K": K0, "answer": ans0, "eval": eval0
        })

        def _accepted(ej: Dict[str, Any]) -> bool:
            if ej.get("verdict", "reject") == "accept":
                return True
            try:
                return float(ej.get("score", 0)) >= 0.7
            except Exception:
                return False

        if _accepted(eval0):
            final_answer = ans0
        else:
            refl1 = ds_client.reflect(question, global_caption, ans0, eval0)
            refined_query = refl1.get("refined_query", question)

            scorer_dyn = Siglip2Scorer(args.siglip_ckpt, device=args.device, dtype=torch.float16)
            q_emb = scorer_dyn.emb_text(refined_query)
            v_emb = scorer_dyn.emb_imgs(frames0, bs=args.siglip_bs)
            idx32_local = mmr_select(v_emb, q_emb, K=seq[1], lam=0.5)
            idx32 = [idx0[i] for i in idx32_local]
            frames1 = [pool_frames[i] for i in idx32]
            save_topk_frames(args.save_frames_root, video_path, f"dynamic_it2_k{seq[1]}", frames1, idx32, None)

            ans1 = bagel_qa_on_frames(frames1, question)
            eval1 = qwen_client.eval_answer(question, global_caption, ans1)
            trace["rounds"].append({
                "type": "dynamic", "iter": 2, "K": len(frames1), "answer": ans1, "eval": eval1
            })

            if _accepted(eval1):
                final_answer = ans1
            else:
                refl2 = ds_client.reflect(question, global_caption, ans1, eval1)
                refined_query = refl2.get("refined_query", refined_query)

                q_emb2 = scorer_dyn.emb_text(refined_query)
                v_emb2 = scorer_dyn.emb_imgs(frames1, bs=args.siglip_bs)
                idx16_local = mmr_select(v_emb2, q_emb2, K=seq[2], lam=0.5)
                idx16 = [idx32[i] for i in idx16_local]
                frames2 = [pool_frames[i] for i in idx16]
                save_topk_frames(args.save_frames_root, video_path, f"dynamic_it3_k{seq[2]}", frames2, idx16, None)

                ans2 = bagel_qa_on_frames(frames2, question)
                eval2 = qwen_client.eval_answer(question, global_caption, ans2)
                trace["rounds"].append({
                    "type": "dynamic", "iter": 3, "K": len(frames2), "answer": ans2, "eval": eval2
                })

                if not _accepted(eval2):
                    fallback = qwen_client.answer_from_global(question, global_caption).strip()
                    if fallback == "" or ("not enough" in fallback.lower() or "insufficient" in fallback.lower()):
                        final_answer = ans0
                        trace["fallback"] = {"reason": "final_score_below_0.7_and_global_not_enough",
                                             "answer_from_qwen": fallback}
                    else:
                        final_answer = fallback
                        trace["fallback"] = {"reason": "final_score_below_0.7", "answer_from_qwen": fallback}
                else:
                    final_answer = ans2

    trace["qtype_final"] = qtype
    trace["final_answer"] = final_answer
    return final_answer, trace


def find_video_by_id(video_dir: str, vid: int, exts: List[str]) -> Optional[str]:
    base = f"video{vid}"
    for ext in exts:
        p = Path(video_dir) / f"{base}{ext}"
        if p.exists():
            return str(p.resolve())
    return None


def main():
    args = parse_args()
    set_seed(args.seed)

    with open(args.gt_file, "r", encoding="utf-8") as f:
        gt_all = json.load(f)

    subset = []
    for item in gt_all:
        if not all(k in item for k in ("video_id", "question", "answer")):
            continue
        try:
            vid = int(str(item["video_id"]).strip())
        except Exception:
            continue
        if args.id_from <= vid <= args.id_to:
            subset.append(item)
    if len(subset) == 0:
        return

    bagel = load_bagel_and_inferencer(args.bagel_model_path, args.max_mem_per_gpu, args.offload_folder)
    ds, qwen = _make_reflection_clients(args.deepseek_api_key)

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.output_name}.json")
    exts = [e if e.startswith(".") else f".{e}" for e in args.video_exts]
    results = []

    for item in subset:
        vid = int(item["video_id"])
        q = item["question"]
        a = item["answer"]
        qid = item.get("id", vid)

        video_path = find_video_by_id(args.video_dir, vid, exts)
        if not video_path:
            results.append({"id": qid, "video_id": vid, "question": q, "answer": a, "pred": "", "trace_path": ""})
            continue

        pred, trace = reflexion_answer_one(video_path, q, bagel, ds, qwen, args)

        base = os.path.splitext(os.path.basename(video_path))[0]
        trace_path = os.path.join(args.output_dir, f"{base}_reflexion.json")
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, ensure_ascii=False, indent=2)

        results.append({
            "id": qid,
            "video_id": vid,
            "question": q,
            "answer": a,
            "pred": pred,
            "trace_path": trace_path
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
