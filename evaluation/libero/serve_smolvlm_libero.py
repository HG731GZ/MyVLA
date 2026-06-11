#!/usr/bin/env python3
"""
SimVLA LIBERO Policy Server (WebSocket)

A WebSocket-based policy server for LIBERO evaluation:
- Uses msgpack_numpy serialization for efficient data transfer
- Sends server metadata on connection
- Receives: observation/image, observation/wrist_image, observation/state, prompt
- Returns: {"actions": [...]}

State format (8D): [ee_pos(3), axis_angle(3), gripper_qpos(2)]
Action format (7D): [delta_xyz(3), delta_axisangle(3), gripper_cmd(1)]
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import traceback
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import websockets

try:
    import msgpack
    import msgpack_numpy
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False
    print("Warning: msgpack_numpy not installed, using JSON fallback")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.modeling_smolvlm_vla import SmolVLMVLA
from models.processing_smolvlm_vla import SmolVLMVLAProcessor
from models.configuration_smolvlm_vla import SmolVLMVLAConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SMOLVLM_MODEL = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Global state
model: Optional[SmolVLMVLA] = None
processor: Optional[SmolVLMVLAProcessor] = None
image_transform = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configuration
CONFIG = {
    "state_dim": 8,
    "action_dim": 7,
    "action_horizon": 10,
    "denoising_steps": 10,
    "image_size": 384,
}


def _looks_like_local_path(value: str) -> bool:
    return value.startswith((".", "/", "~")) or os.path.exists(os.path.expanduser(value))


def _resolve_existing_path(value: str, bases) -> Optional[Path]:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [base / path for base in bases]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _resolve_smolvlm_model_path(
    checkpoint_path: Path,
    configured_path: str,
    override_path: Optional[str],
) -> str:
    explicit_path = override_path or os.environ.get("SMOLVLM_MODEL_PATH")
    bases = (Path.cwd(), PROJECT_ROOT, checkpoint_path.parent)

    if explicit_path:
        resolved = _resolve_existing_path(explicit_path, bases)
        if resolved is not None:
            return str(resolved)
        if _looks_like_local_path(explicit_path):
            raise FileNotFoundError(
                f"SmolVLM model path does not exist: {explicit_path}. "
                "Pass --smolvlm_model or set SMOLVLM_MODEL_PATH."
            )
        return explicit_path

    resolved = _resolve_existing_path(configured_path, bases)
    if resolved is not None:
        return str(resolved)
    if configured_path and not _looks_like_local_path(configured_path):
        return configured_path

    logger.warning(
        "Checkpoint SmolVLM path %r is not available on this machine; "
        "falling back to %s. Set SMOLVLM_MODEL_PATH to use a local copy.",
        configured_path,
        DEFAULT_SMOLVLM_MODEL,
    )
    return DEFAULT_SMOLVLM_MODEL


def _resolve_norm_stats_path(
    checkpoint_path: Path,
    override_path: Optional[str],
) -> Path:
    requested_path = override_path or os.environ.get("NORM_STATS_PATH")
    bases = (Path.cwd(), PROJECT_ROOT, checkpoint_path.parent)

    if requested_path:
        resolved = _resolve_existing_path(requested_path, bases)
        if resolved is None:
            raise FileNotFoundError(f"Norm stats file does not exist: {requested_path}")
        return resolved

    training_config_path = checkpoint_path / "training_config.json"
    if training_config_path.is_file():
        with training_config_path.open() as f:
            training_config = json.load(f)
        configured_path = training_config.get("norm_stats_path")
        if configured_path:
            resolved = _resolve_existing_path(configured_path, bases)
            if resolved is not None:
                return resolved

    raise FileNotFoundError(
        "Normalization stats are required for LIBERO evaluation. Pass "
        "--norm_stats, set NORM_STATS_PATH, or keep a valid norm_stats_path "
        "in the checkpoint training_config.json."
    )


def _build_image_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize(
            (image_size, image_size),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def _autocast_context():
    if model is None or torch.device(device).type != "cuda":
        return nullcontext()
    vlm_dtype = next(model.vlm.parameters()).dtype
    if vlm_dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=vlm_dtype)
    return nullcontext()


def load_model(
    checkpoint_path: str,
    norm_stats_path: str = None,
    smolvlm_model_path: str = None,
    denoising_steps: int = 10,
):
    """Load SimVLA model and processor."""
    global model, processor, image_transform
    
    logger.info(f"Loading SimVLA from {checkpoint_path}...")

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_path}")

    config = SmolVLMVLAConfig.from_pretrained(checkpoint_path)
    resolved_smolvlm_path = _resolve_smolvlm_model_path(
        checkpoint_path,
        config.smolvlm_model_path,
        smolvlm_model_path,
    )
    config.smolvlm_model_path = resolved_smolvlm_path

    model = SmolVLMVLA(config)
    safetensors_path = checkpoint_path / "model.safetensors"
    pytorch_path = checkpoint_path / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path), device="cpu")
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No model.safetensors or pytorch_model.bin found in {checkpoint_path}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint is incompatible with the current Cross-Self network structure."
        ) from exc
    del state_dict

    if model.action_mode != "libero_joint":
        raise ValueError(
            f"LIBERO evaluation requires action_mode='libero_joint', got {model.action_mode!r}."
        )
    if model.num_views < 2:
        raise ValueError(f"Model num_views={model.num_views} cannot hold both LIBERO views")

    resolved_norm_stats = _resolve_norm_stats_path(checkpoint_path, norm_stats_path)
    logger.info(f"Loading norm stats from: {resolved_norm_stats}")
    model.action_space.load_norm_stats(str(resolved_norm_stats))
    
    processor = SmolVLMVLAProcessor(
        smolvlm_model_path=resolved_smolvlm_path,
        num_views=model.num_views,
        image_size=model.image_size,
        language_max_length=getattr(model.config, "language_max_length", 96),
    )

    CONFIG["action_horizon"] = int(model.num_actions)
    CONFIG["action_dim"] = int(model.action_space.dim_action)
    CONFIG["state_dim"] = int(model.action_space.dim_proprio)
    CONFIG["image_size"] = int(model.image_size)
    CONFIG["denoising_steps"] = max(1, int(denoising_steps))
    image_transform = _build_image_transform(model.image_size)

    model = model.to(device)
    model.eval()

    logger.info(
        "Model loaded: device=%s, views=%d, image=%dx%d, action_horizon=%d, denoising_steps=%d",
        device,
        model.num_views,
        CONFIG["image_size"],
        CONFIG["image_size"],
        CONFIG["action_horizon"],
        CONFIG["denoising_steps"],
    )


def preprocess_images(image0: np.ndarray, image1: np.ndarray):
    """Preprocess images to model input format."""
    if model is None or image_transform is None:
        raise RuntimeError("Model must be loaded before preprocessing images")

    def prepare_image(image: np.ndarray, name: str) -> torch.Tensor:
        image = np.asarray(image)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{name} must have shape [H, W, 3], got {image.shape}")
        image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
        return image_transform(Image.fromarray(image))

    img0_t = prepare_image(image0, "observation/image")
    img1_t = prepare_image(image1, "observation/wrist_image")

    views = [img0_t, img1_t]
    views.extend(torch.zeros_like(img0_t) for _ in range(model.num_views - 2))
    images = torch.stack(views, dim=0)
    image_mask = torch.zeros(1, model.num_views, dtype=torch.bool)
    image_mask[:, :2] = True
    
    return images.unsqueeze(0), image_mask


def decode_numpy(obj):
    """Decode numpy array from msgpack_numpy dict format."""
    if isinstance(obj, dict):
        if b'__ndarray__' in obj or '__ndarray__' in obj:
            data_key = b'data' if b'data' in obj else 'data'
            dtype_key = b'dtype' if b'dtype' in obj else 'dtype'
            shape_key = b'shape' if b'shape' in obj else 'shape'
            
            data = obj[data_key]
            dtype_str = obj[dtype_key]
            shape = obj[shape_key]
            
            if isinstance(dtype_str, bytes):
                dtype_str = dtype_str.decode()
            
            if shape and isinstance(shape[0], bytes):
                shape = tuple(int(s) for s in shape)
            else:
                shape = tuple(shape)
            
            return np.frombuffer(data, dtype=np.dtype(dtype_str)).reshape(shape)
    return obj


def infer(observation: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference on a single observation."""
    global model, processor

    if model is None or processor is None:
        raise RuntimeError("Model must be loaded before inference")

    image0 = decode_numpy(observation.get("observation/image"))
    image1 = decode_numpy(observation.get("observation/wrist_image"))
    state = decode_numpy(observation.get("observation/state"))
    prompt = observation.get("prompt", "")

    if image0 is None or image1 is None:
        raise ValueError("Both agent-view and wrist-view images are required")
    if state is None:
        raise ValueError("observation/state is required")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A non-empty prompt is required")

    state = np.array(state, dtype=np.float32, copy=True).reshape(-1)
    if state.size != CONFIG["state_dim"]:
        raise ValueError(
            f"observation/state must contain {CONFIG['state_dim']} values, got {state.size}"
        )
    if not np.isfinite(state).all():
        raise ValueError("observation/state contains non-finite values")

    images, image_mask = preprocess_images(image0, image1)
    images = images.to(device)
    image_mask = image_mask.to(device)
    lang = {
        key: value.to(device)
        for key, value in processor.encode_language([prompt]).items()
    }
    proprio_tensor = torch.from_numpy(state).unsqueeze(0).to(device)

    with torch.inference_mode(), _autocast_context():
        actions = model.generate_actions(
            input_ids=lang["input_ids"],
            text_attention_mask=lang["text_attention_mask"],
            image_input=images,
            image_mask=image_mask,
            proprio=proprio_tensor,
            steps=CONFIG["denoising_steps"],
        )

    actions = actions.float().cpu().numpy()[0]
    expected_shape = (CONFIG["action_horizon"], CONFIG["action_dim"])
    if actions.shape != expected_shape:
        raise RuntimeError(f"Model returned actions with shape {actions.shape}, expected {expected_shape}")
    if not np.isfinite(actions).all():
        raise RuntimeError("Model returned non-finite actions")
    return {"actions": actions}


async def handle_connection(websocket, path=None):
    """Handle a WebSocket connection."""
    logger.info(f"Connection from {websocket.remote_address} opened")
    
    try:
        # Send server metadata on connection
        metadata = {
            "model": "SimVLA",
            "action_dim": CONFIG["action_dim"],
            "action_horizon": CONFIG["action_horizon"],
            "denoising_steps": CONFIG["denoising_steps"],
            "image_size": CONFIG["image_size"],
        }
        if HAS_MSGPACK:
            await websocket.send(msgpack_numpy.packb(metadata, use_bin_type=True))
        else:
            import json
            await websocket.send(json.dumps(metadata))
        
        # Process requests
        async for message in websocket:
            try:
                # Parse request
                if HAS_MSGPACK and isinstance(message, bytes):
                    request = msgpack_numpy.unpackb(message, raw=False)
                else:
                    import json
                    request = json.loads(message)
                
                # Run inference
                result = infer(request)
                
                # Send response (convert numpy to list for compatibility)
                actions = result["actions"]
                if isinstance(actions, np.ndarray):
                    actions = actions.tolist()
                
                response_data = {"actions": actions}
                
                if HAS_MSGPACK:
                    import msgpack
                    response = msgpack.packb(response_data, use_bin_type=True)
                else:
                    import json
                    response = json.dumps(response_data)
                
                await websocket.send(response)
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                traceback.print_exc()
                error_data = {"error": str(e)}
                if HAS_MSGPACK:
                    await websocket.send(msgpack.packb(error_data, use_bin_type=True))
                else:
                    await websocket.send(json.dumps(error_data))
                
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        logger.info(f"Connection from {websocket.remote_address} closed")


async def serve(host: str, port: int):
    """Start the WebSocket server."""
    logger.info(f"Creating SimVLA server (host: {host}, port: {port})")
    
    async with websockets.serve(handle_connection, host, port, max_size=None, compression=None):
        logger.info(f"SimVLA server listening on {host}:{port}")
        await asyncio.Future()


def main():
    parser = argparse.ArgumentParser(description="SimVLA LIBERO Server (WebSocket)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SimVLA checkpoint")
    parser.add_argument("--norm_stats", type=str, default=None,
                        help="Path to normalization stats JSON; inferred from training_config.json when omitted")
    parser.add_argument("--smolvlm_model", type=str, default=None,
                        help="SmolVLM model path or HuggingFace repo; can also use SMOLVLM_MODEL_PATH")
    parser.add_argument("--denoising_steps", type=int, default=10,
                        help="Number of Euler flow-matching denoising steps")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    
    args = parser.parse_args()
    
    if not HAS_MSGPACK:
        logger.warning("msgpack_numpy not installed! Install with: pip install msgpack-numpy")
    
    load_model(
        args.checkpoint,
        args.norm_stats,
        args.smolvlm_model,
        args.denoising_steps,
    )
    
    logger.info(f"Starting SimVLA server on {args.host}:{args.port}")
    logger.info(f"  Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
    logger.info(f"  Action horizon: {CONFIG['action_horizon']}")
    logger.info(f"  Denoising steps: {CONFIG['denoising_steps']}")
    
    asyncio.run(serve(args.host, args.port))


if __name__ == "__main__":
    main()
