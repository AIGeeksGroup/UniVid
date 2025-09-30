import os
import sys
import torch
import logging
import argparse
from pathlib import Path
from PIL import Image
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(CURRENT_DIR.parent))
if CURRENT_DIR.name != 'UniVid':
    univid_dir = CURRENT_DIR.parent if CURRENT_DIR.parent.name == 'UniVid' else CURRENT_DIR.parent / 'UniVid'
    if univid_dir.exists():
        sys.path.insert(0, str(univid_dir))

from model_pipeline import (
    CrossAttentionConfig,
    CrossAttentionFusionPipeline
)

try:
    sys.path.append(str(CURRENT_DIR / "Wan22"))
    from Wan22.wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
    PROMPT_EXTEND_AVAILABLE = True
except ImportError:
    PROMPT_EXTEND_AVAILABLE = False
    print("⚠️ Prompt extend not available. Install Wan2.2 dependencies to enable.")

@dataclass
class VideoGenerationConfig:

    bagel_model_path: str = "your_model_path"
    wan_model_path: str = "your_model_path"

    use_lora: bool = False  
    lora_checkpoint_path: str = "your_lora_path"  

    # multi-gpu
    bagel_gpu: int = 0      
    wan_gpu: int = 1
    cross_attn_gpu: int = 2


    video_length: int = 121  
    video_fps: int = 24
    video_size: Tuple[int, int] = (1280, 704)


    num_inference_steps: int = 50
    guidance_scale: float = 5.0
    sample_shift: float = 5.0

    # Cross Attention
    guidance_strength: float = 1.0   
    bagel_sequence_length: int = 256
    wan_text_length: int = 512

    negative_prompt: str = (
      "distorted, deformed, warped, bent, twisted, morphing, "
      "inconsistent geometry, unstable shapes, melting objects, "
      "flickering, jittering, temporal artifacts, "
      "bad anatomy, incorrect proportions, asymmetric features"
  )

    use_dynamic_text_weight: bool = True      
    text_weight_max: float = 1.5
    text_weight_min: float = 1.0
    text_weight_schedule: str = "cosine"
    text_weight_transition_ratio: float = 0.4
    total_sampling_steps: int = 50

    output_dir: str = "your_output_dir"
    save_video_mp4: bool = True
    video_codec: str = "h264"  
    video_bitrate: str = "10M"
    video_preset: str = "slow"

    use_bfloat16: bool = True
    enable_autocast: bool = True
    skip_t5_loading: bool = False  
    seed: int = 42


    verbose: bool = True


    use_prompt_extend: bool = False
    prompt_extend_method: str = "local_qwen" 
    prompt_extend_model: str = "your_model_name"  
    prompt_extend_target_lang: str = "en"  


class HighQualityVideoGenerator:

    
    def __init__(self, config: VideoGenerationConfig):
        self.config = config
        self.setup_logging()
        self.setup_prompt_expander()
        self.setup_pipeline()
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO if self.config.verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_prompt_expander(self):
        """设置Prompt Expander"""
        self.prompt_expander = None
        if self.config.use_prompt_extend and PROMPT_EXTEND_AVAILABLE:
            try:
                if self.config.prompt_extend_method == "dashscope":
                    self.prompt_expander = DashScopePromptExpander(
                        model_name=self.config.prompt_extend_model,
                        model_class="chat"
                    )
                    self.logger.info(f"✅ DashScope Prompt Expander initialized: {self.config.prompt_extend_model}")
                elif self.config.prompt_extend_method == "local_qwen":
                    self.prompt_expander = QwenPromptExpander(
                        model_name=self.config.prompt_extend_model,
                        dtype="bf16"
                    )
                    self.logger.info(f"✅ Local Qwen Prompt Expander initialized: {self.config.prompt_extend_model}")
                else:
                    self.logger.warning(f"⚠️ Unknown prompt extend method: {self.config.prompt_extend_method}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize prompt expander: {e}")
                self.prompt_expander = None
        elif self.config.use_prompt_extend:
            self.logger.warning("⚠️ Prompt extend requested but not available")
    
    def setup_pipeline(self):
        self.logger.info("🚀 Initializing High-Quality Video Generation Pipeline...")
        if self.config.use_lora:
            self.logger.info(f"🎯 LoRA Enabled: Loading from {self.config.lora_checkpoint_path}")
        self.logger.info("✨ Feature: Dynamic Text Weight Scheduling Enabled")

        ca_config = CrossAttentionConfig(

            bagel_model_path=self.config.bagel_model_path,
            wan_model_path=self.config.wan_model_path,


            bagel_gpu=self.config.bagel_gpu,
            wan_gpu=self.config.wan_gpu,
            cross_attn_gpu=self.config.cross_attn_gpu,


            video_length=self.config.video_length,   
            video_fps=self.config.video_fps,         
            video_size=self.config.video_size,


            fusion_mode="context_replacement",
            guidance_strength=self.config.guidance_strength,  
            wan_text_length=self.config.wan_text_length,


            output_dir=self.config.output_dir,
            save_video_mp4=self.config.save_video_mp4,

            use_lora=self.config.use_lora,
            lora_rank=8,  
            lora_alpha=16,
            lora_dropout=0.0,
            lora_target_strategy="your_lora_target_strategy",
            
            use_bfloat16=self.config.use_bfloat16,
            enable_autocast=self.config.enable_autocast,
            skip_t5_loading=self.config.skip_t5_loading,  
            
            use_dynamic_text_weight=self.config.use_dynamic_text_weight,  
            text_weight_max=self.config.text_weight_max,
            text_weight_min=self.config.text_weight_min,
            text_weight_schedule=self.config.text_weight_schedule,
            text_weight_transition_ratio=self.config.text_weight_transition_ratio,
            total_sampling_steps=self.config.total_sampling_steps,


            enable_bagel_extraction=True,
            enable_wan_injection=True,
            freeze_bagel=True,
            freeze_wan_vae=True,
            freeze_t5=True,
            train_wan_dit=False,
            train_cross_attn=False,
        )
        
        self.pipeline = CrossAttentionFusionPipeline(ca_config)

        if self.config.use_lora:
            self.load_lora_weights()


        if self.config.use_dynamic_text_weight:
            self.logger.info(f"📈 Text Weight Schedule: {self.config.text_weight_schedule}")
            self.logger.info(f"📊 Weight Range: {self.config.text_weight_max} → {self.config.text_weight_min}")
            self.logger.info(f"⏱️ Transition Ratio: {self.config.text_weight_transition_ratio * 100}%")

        self.logger.info("✅ Pipeline initialized successfully!")

    def load_lora_weights(self):
        from pathlib import Path
        checkpoint_path = Path(self.config.lora_checkpoint_path)

        if not checkpoint_path.exists():
            self.logger.error(f"❌ LoRA checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"LoRA checkpoint not found: {checkpoint_path}")

        try:
            if hasattr(self.pipeline, 'lora_manager') and self.pipeline.lora_manager:
                self.logger.info(f"📦 Loading LoRA weights from: {checkpoint_path}")

                self.pipeline.lora_manager.load_lora_weights(
                    str(checkpoint_path),
                    self.pipeline.dit_model  
                )
                self.logger.info("✅ LoRA weights loaded via LoRA manager")

            projector_path = checkpoint_path / "training_state.pt"
            if projector_path.exists():
                self.logger.info(f"📦 Loading Context Projector from: {projector_path}")
                state = torch.load(projector_path, map_location=f"cuda:{self.config.cross_attn_gpu}")

                if 'context_projector' in state and hasattr(self.pipeline, 'context_projector'):
                    self.pipeline.context_projector.load_state_dict(state['context_projector'])
                    self.logger.info("✅ Context Projector loaded")
                else:
                    self.logger.warning("⚠️ Context Projector not found in checkpoint")

            if hasattr(self.pipeline, 'dit_model'):
                lora_params_count = 0
                all_params = []
                for name, param in self.pipeline.dit_model.named_parameters():

                    if ('lora' in name.lower() or 'lora_A' in name or 'lora_B' in name):
                        lora_params_count += 1
                        all_params.append(name)

                if lora_params_count > 0:
                    self.logger.info(f"✅ LoRA verification: {lora_params_count} LoRA parameters active")
                    for i, name in enumerate(all_params[:3]):
                        self.logger.info(f"   Example {i+1}: {name}")
                else:
                    self.logger.warning("⚠️ No LoRA parameters found after loading")
                    param_patterns = set()
                    for name, _ in self.pipeline.dit_model.named_parameters():
                        parts = name.split('.')
                        if len(parts) > 2:
                            pattern = '.'.join(parts[:3])
                            param_patterns.add(pattern)
                    self.logger.info(f"   Available param patterns: {list(param_patterns)[:5]}")

            self.logger.info("✅ All LoRA components loaded successfully!")

        except Exception as e:
            self.logger.error(f"❌ Failed to load LoRA weights: {e}")
            raise

    def generate_text_to_video(
        self,
        prompt: str,
        output_name: Optional[str] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[str]]:

        self.logger.info("🎬 Generating Text-to-Video ...")
        self.logger.info(f"📝 Original Prompt: {prompt}")

        # Prompt Extend
        if self.prompt_expander is not None:
            try:
                self.logger.info("🔄 Extending prompt...")
                system_prompt = "You are a professional video prompt engineer. Enhance the given prompt to create a more detailed and vivid video description."
                extended_prompt = self.prompt_expander(
                    prompt,
                    system_prompt,
                    seed=self.config.seed,
                    tar_lang=self.config.prompt_extend_target_lang
                )
                self.logger.info(f"✨ Extended Prompt: {extended_prompt[:200]}...")
                prompt = extended_prompt
            except Exception as e:
                self.logger.warning(f"⚠️ Prompt extension failed: {e}, using original prompt")

        if self.config.use_dynamic_text_weight:
            self.logger.info(f"📈 : Dynamic text weight enabled ({self.config.text_weight_max}→{self.config.text_weight_min})")

        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"t2v_{timestamp}"
        
        generation_kwargs = {

            "steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "frames": self.config.video_length,
            "size": self.config.video_size,
            "shift": self.config.sample_shift,
            "seed": self.config.seed,
        }
        

        try:
            video_tensor, video_path = self.pipeline.generate_video_with_bagel_context(
                text=prompt,
                **generation_kwargs
            )
            
            if video_tensor is not None:
                if self.config.save_video_mp4:
                    video_path = self.save_high_quality_video(
                        video_tensor, 
                        output_name,
                        prompt
                    )
                
                self.logger.info(f"✅ Video generated successfully!")
                self.logger.info(f"📁 Saved to: {video_path}")
                return video_tensor, video_path
            else:
                self.logger.error("❌ Video generation failed!")
                return None, None
                
        except Exception as e:
            self.logger.error(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def generate_image_to_video(
        self,
        image_path: str,
        prompt: str,
        output_name: Optional[str] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[str]]:
        
        self.logger.info("🎬 Generating Image-to-Video ()...")
        self.logger.info(f"🖼️ Image: {image_path}")
        self.logger.info(f"📝 Prompt: {prompt}")
        if self.config.use_dynamic_text_weight:
            self.logger.info(f"📈 : Dynamic text weight enabled ({self.config.text_weight_max}→{self.config.text_weight_min})")
        
        # 加载图像
        if not os.path.exists(image_path):
            self.logger.error(f"❌ Image not found: {image_path}")
            return None, None
        
        try:
            image = Image.open(image_path).convert("RGB")
            self.logger.info(f"   Image size: {image.size}")
        except Exception as e:
            self.logger.error(f"❌ Failed to load image: {e}")
            return None, None
        
        # 设置输出文件名
        if output_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"i2v_{timestamp}"
        
        generation_kwargs = {
            "steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
            "frames": self.config.video_length,
            "size": self.config.video_size,
            "shift": self.config.sample_shift,
            "seed": self.config.seed,
        }

        try:
            video_tensor, video_path = self.pipeline.generate_video_with_bagel_context(
                text=prompt,
                image=image,
                **generation_kwargs
            )
            
            if video_tensor is not None:
                if self.config.save_video_mp4:
                    video_path = self.save_high_quality_video(
                        video_tensor,
                        output_name,
                        f"{prompt} (from image)"
                    )
                
                self.logger.info(f"✅ Video generated successfully!")
                self.logger.info(f"📁 Saved to: {video_path}")
                return video_tensor, video_path
            else:
                self.logger.error("❌ Video generation failed!")
                return None, None
                
        except Exception as e:
            self.logger.error(f"❌ Generation error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def save_high_quality_video(
        self,
        video_tensor: torch.Tensor,
        output_name: str,
        description: str = ""
    ) -> str:
        
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.logger.error("❌ OpenCV not available, falling back to tensor save")
            return self._save_tensor_fallback(video_tensor, output_name)
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{output_name}.mp4"
        
        try:
            if video_tensor.dim() == 5:  # [B, C, T, H, W]
                video_tensor = video_tensor[0]  
            
            if video_tensor.dim() == 4 and video_tensor.shape[0] == 3:  # [C, T, H, W]
                video_tensor = video_tensor.permute(1, 2, 3, 0)  # [T, H, W, C]
            elif video_tensor.dim() == 4:  # [T, H, W, C]
                pass  
            
            video_numpy = video_tensor.cpu().numpy()

            if video_numpy.min() >= -1.5 and video_numpy.max() <= 1.5:
                video_numpy = (video_numpy + 1) / 2  # [-1, 1] -> [0, 1]
                video_numpy = np.clip(video_numpy * 255, 0, 255).astype(np.uint8)
            elif video_numpy.max() <= 1.0:
                video_numpy = np.clip(video_numpy * 255, 0, 255).astype(np.uint8)
            else:
                video_numpy = np.clip(video_numpy, 0, 255).astype(np.uint8)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

            T, H, W, C = video_numpy.shape
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                self.config.video_fps,
                (W, H)
            )
            
            for frame_idx in range(T):
                frame = video_numpy[frame_idx]
                # RGB to BGR for OpenCV
                if C == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                writer.write(frame_bgr)
            
            writer.release()
 
            self._improve_video_quality(output_path)

            self._save_video_metadata(output_path, output_name, description, T, H, W)
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"❌ Video saving failed: {e}")
            return self._save_tensor_fallback(video_tensor, output_name)
    
    def _improve_video_quality(self, video_path: Path):
        try:
            temp_path = video_path.with_suffix('.temp.mp4')
            ffmpeg_cmd = (
                f"ffmpeg -i {video_path} "
                f"-c:v {self.config.video_codec} "
                f"-preset {self.config.video_preset} "
                f"-b:v {self.config.video_bitrate} "
                f"-pix_fmt yuv420p "
                f"-movflags +faststart "
                f"{temp_path} -y"
            )
            
            result = os.system(ffmpeg_cmd + " > /dev/null 2>&1")
            
            if result == 0 and temp_path.exists():
                video_path.unlink()
                temp_path.rename(video_path)
                self.logger.info(f"✅ Video quality improved with ffmpeg")
            else:
                if temp_path.exists():
                    temp_path.unlink()
                self.logger.info(f"⚠️ ffmpeg improvement failed, using original")
        except Exception as e:
            self.logger.warning(f"⚠️ ffmpeg improvement failed: {e}")
    
    def _save_tensor_fallback(self, video_tensor: torch.Tensor, output_name: str) -> str:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        tensor_path = output_dir / f"{output_name}.pt"
        torch.save(video_tensor, str(tensor_path))
        self.logger.info(f"💾 Tensor fallback saved: {tensor_path}")
        return str(tensor_path)
    
    def _save_video_metadata(self, video_path: Path, output_name: str, description: str, T: int, H: int, W: int):
        try:
            metadata_path = video_path.with_suffix('.txt')
            with open(metadata_path, 'w') as f:
                f.write(f"Description: {description}\n")
                f.write(f"Frames: {T}\n")
                f.write(f"FPS: {self.config.video_fps}\n")
                f.write(f"Resolution: {W}x{H}\n")
                f.write(f"Duration: {T/self.config.video_fps:.2f} seconds\n")
                f.write(f"Codec: {self.config.video_codec}\n")
                f.write(f"Bitrate: {self.config.video_bitrate}\n")
                f.write(f"Generated: {datetime.now().isoformat()}\n")
            
            self.logger.info(f"📄 Metadata saved: {metadata_path}")
        except Exception as e:
            self.logger.warning(f"⚠️ Metadata save failed: {e}")


def main():
    
    parser = argparse.ArgumentParser(description="High-Quality Video Generation  with Dynamic Text Weight")
    parser.add_argument("--mode", type=str, choices=["t2v", "i2v", "both"], default="both",
                       help="Generation mode: t2v (text-to-video), i2v (image-to-video), or both")
    parser.add_argument("--image", type=str, default="/fs/scratch/PFIN0007/ICLR_2025/UniVid/Wan22/examples/i2v_input.JPG",
                       help="Input image path for i2v mode")
    parser.add_argument("--output_dir", type=str, default="./outputs__high_quality",
                       help="Output directory for generated videos")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=5.0,
                       help="Guidance scale (CFG)")
    parser.add_argument("--bagel_strength", type=float, default=1.0,
                       help="BAGEL fusion strength (0=no BAGEL, 1=full BAGEL)")
    parser.add_argument("--use_lora", action="store_true",
                       help="Use LoRA-enhanced model for generation")
    parser.add_argument("--lora_path", type=str, default="./lora_checkpoints/best",
                       help="Path to LoRA checkpoint")
    parser.add_argument("--video_length", type=int, default=None,
                       help="Video length in frames (16 for fast, 61 for HD)")
    parser.add_argument("--video_size", type=str, default='hd',
                       help="Video size: 'training' (512x320) or 'hd' (1280x704)")
    parser.add_argument("--disable_dynamic_weight", action="store_true",
                       help="Disable  dynamic text weight scheduling")
    parser.add_argument("--text_weight_max", type=float, default=1.5,
                       help="Maximum text weight in early phase")
    parser.add_argument("--text_weight_min", type=float, default=1.0,
                       help="Minimum text weight in late phase")
    parser.add_argument("--weight_schedule", type=str, default="cosine",
                       choices=["linear", "cosine", "exponential"],
                       help="Weight scheduling strategy")
    parser.add_argument("--transition_ratio", type=float, default=0.4,
                       help="Transition phase ratio (0-1)")
    parser.add_argument("--prompt", type=str, default=None,
                       help="Text prompt for generation (overrides default)")
    parser.add_argument("--use_prompt_extend", action="store_true",
                       help="Use prompt extension to enhance the input prompt")
    parser.add_argument("--prompt_extend_method", type=str, default="local_qwen",
                       choices=["dashscope", "local_qwen"],
                       help="Prompt extension method")
    parser.add_argument("--prompt_extend_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                       help="Model for prompt extension")
    parser.add_argument("--prompt_extend_target_lang", type=str, default="en",
                       help="Target language for prompt extension")

    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("🎬 HIGH-QUALITY VIDEO GENERATION ")
    print("🚀 BAGEL + Wan2.2 Cross Attention Fusion")
    if args.use_lora:
        print(f"🎯 LoRA Enhanced: {args.lora_path}")
    print("✨  Feature: Dynamic Text Weight Scheduling")
    print("="*80 + "\n")

    if args.video_length is not None:
        video_length = args.video_length
    else:
        video_length = 121  
    video_size = (1280, 704)  
    print(f"✅ Using HD resolution: {video_size}")

    if args.video_size:
        print(f"   Note: video_size parameter '{args.video_size}' ignored, using unified HD resolution")

    config = VideoGenerationConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        use_lora=args.use_lora,
        lora_checkpoint_path=args.lora_path,
        video_length=video_length,
        video_size=video_size,
        use_dynamic_text_weight=not args.disable_dynamic_weight,
        text_weight_max=args.text_weight_max,  
        text_weight_min=args.text_weight_min,
        text_weight_schedule=args.weight_schedule,
        text_weight_transition_ratio=args.transition_ratio,  
        total_sampling_steps=args.steps,
        guidance_strength=args.bagel_strength, 
        use_prompt_extend=args.use_prompt_extend,
        prompt_extend_method=args.prompt_extend_method,
        prompt_extend_model=args.prompt_extend_model,
        prompt_extend_target_lang=args.prompt_extend_target_lang,
        skip_t5_loading=False  
    )

    if config.use_dynamic_text_weight:
        print("📈  Dynamic Text Weight Configuration:")
        print(f"   • Schedule: {config.text_weight_schedule}")
        print(f"   • Weight Range: {config.text_weight_max} → {config.text_weight_min}")
        print(f"   • Transition: {config.text_weight_transition_ratio * 100}% of steps")
        print()

    try:
        generator = HighQualityVideoGenerator(config)
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        import traceback
        traceback.print_exc()
        return

    if args.mode in ["t2v", "both"]:
        print("\n" + "-"*40)
        print("🎯 Text-to-Video Generation")
        print("-"*40)

        if args.prompt:
            t2v_prompt = args.prompt
        else:

            import sys
            if not sys.stdin.isatty():
                t2v_prompt = sys.stdin.read().strip()
            else:
                t2v_prompt = (
                    "Two anthropomorphic cats in comfy boxing gear and bright gloves "
                    "fight intensely on a spotlighted stage."
                )

        print(f"Prompt: {t2v_prompt}")
        print("\nGenerating...")
 
        safe_name = t2v_prompt.replace(' ', '_').replace(',', '').replace('.', '')
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-')[:80]

        video_tensor, video_path = generator.generate_text_to_video(
            prompt=t2v_prompt,
            output_name=safe_name
        )
        
        if video_path:
            print(f"✅ Success! Video saved to: {video_path}")
        else:
            print("❌ Text-to-Video generation failed")
    if args.mode in ["i2v", "both"]:
        print("\n" + "-"*40)
        print("🎯 Image-to-Video Generation")
        print("-"*40)
        
        i2v_prompt = (
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. "
            "The fluffy-furred feline gazes directly at the camera with a relaxed expression. "
            "Blurred beach scenery forms the background featuring crystal-clear waters, "
            "distant green hills, and a blue sky dotted with white clouds. "
            "The cat assumes a naturally relaxed posture, as if savoring the sea breeze "
            "and warm sunlight. A close-up shot highlights the feline's intricate details "
            "and the refreshing atmosphere of the seaside."
        )
        
        print(f"Image: {args.image}")
        print(f"Prompt: {i2v_prompt[:100]}...")
        print("\nGenerating...")

        image_path = Path(args.image)
        if not image_path.exists():
            print(f"⚠️ Image not found at {args.image}")
            print("Creating a sample image...")

            from PIL import Image, ImageDraw
            img = Image.new('RGB', (512, 512), color='skyblue')
            draw = ImageDraw.Draw(img)
            draw.ellipse([156, 156, 356, 356], fill='white')  

            os.makedirs("examples", exist_ok=True)
            image_path = Path("/fs/scratch/PFIN0007/ICLR_2025/UniVid/Wan22/examples/i2v_input.JPG")
            img.save(image_path)
            print(f"Sample image saved to: {image_path}")
        
        video_tensor, video_path = generator.generate_image_to_video(
            image_path=str(image_path),
            prompt=i2v_prompt,
            output_name="beach_cat_7s"
        )
        
        if video_path:
            print(f"✅ Success! Video saved to: {video_path}")
        else:
            print("❌ Image-to-Video generation failed")
    
    print("\n" + "="*80)
    print("🎉 Generation Complete!")
    print("\n📊 High-Quality Settings Used:")
    print(f"   • Resolution: {config.video_size[0]}x{config.video_size[1]}")
    print(f"   • Frame Rate: {config.video_fps} fps")
    print(f"   • Duration: ~7 seconds ({config.video_length} frames)")
    print(f"   • Inference Steps: {config.num_inference_steps}")
    print(f"   • Guidance Scale: {config.guidance_scale}")
    print(f"   • BAGEL Context Strength: {config.guidance_strength}")
    print(f"   • Video Codec: {config.video_codec}")
    print(f"   • Bitrate: {config.video_bitrate}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()