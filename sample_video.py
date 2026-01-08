import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime
import torch

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from op_tracer import OpAndModuleTracer, EventRecorder

spm_name_map = {
    "scaled_dot_product_attention": "ScaledDotProductAttention",
    "RMSNorm": "RMSNorm",
    "LayerNorm": "LayerNorm",
}

def choose_module_to_trace(sampler):
    if hasattr(sampler, "pipeline"):
        pipeline = sampler.pipeline
        if hasattr(pipeline, "transformer") and isinstance(pipeline.transformer, torch.nn.Module):
            logger.info("Tracing module: pipeline.transformer")
            return pipeline.transformer
        if isinstance(pipeline, torch.nn.Module):
            logger.info("Tracing module: pipeline")
            return pipeline
    if hasattr(sampler, "model") and isinstance(sampler.model, torch.nn.Module):
        logger.info("Tracing module: model")
        return sampler.model
    candidates = [
        (name, module)
        for name, module in sampler.__dict__.items()
        if isinstance(module, torch.nn.Module)
    ]
    if candidates:
        name, module = max(
            candidates, key=lambda kv: sum(p.numel() for p in kv[1].parameters(recurse=True))
        )
        logger.info(f"Tracing module (fallback): {name}")
        return module
    raise RuntimeError("No torch.nn.Module found in sampler to trace.")


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_rank = "LOCAL_RANK" not in os.environ or local_rank == 0

    tracer = None
    recorder = None
    if is_main_rank:
        module_to_trace = choose_module_to_trace(hunyuan_video_sampler)
        recorder = EventRecorder()
        tracer = OpAndModuleTracer(
            module_to_trace,
            recorder,
            spm_name_map=spm_name_map,
            leaf_only=True,
            capture_tensors=True,
            max_tensors_per_event=8,
        ).start()

    # Start sampling
    # TODO: batch inference check
    try:
        outputs = hunyuan_video_sampler.predict(
            prompt=args.prompt, 
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=args.seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale
        )
    finally:
        if tracer is not None:
            tracer.stop()
            trace_path = os.path.join(save_path, "trace.json")
            recorder.save(trace_path)
            logger.info(f"Trace saved to: {trace_path}")
    samples = outputs['samples']
    
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            cur_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, cur_save_path, fps=24)
            logger.info(f'Sample save to: {cur_save_path}')

if __name__ == "__main__":
    main()
