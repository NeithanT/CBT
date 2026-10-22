import os
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE_PATH = os.path.join(ROOT_DIR, "config.json")


@dataclass
class TokenizerConstants:
    special_tokens: List[str] = field(
        default_factory=lambda: ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<THINK>", "</THINK>"]
    )
    digits: List[str] = field(
        default_factory=lambda: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    )
    operators: List[str] = field(
        default_factory=lambda: ["+", "-", "*", "/", "^", "%", "(", ")"]
    )
    separators: List[str] = field(default_factory=lambda: ["=", " ", "\n"])


@dataclass
class ModelConstants:
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    dropout: float = 0.05
    bias: bool = False
    pos_emb_type: str = "rope"
    use_pos_emb: bool = False


@dataclass
class LoopedModelSettings:
    block_size: int = 512
    n_embd: int = 512
    n_head: int = 8
    n_core_layers: int = 2
    train_loops: int = 6
    eval_loops: int = 12
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    deep_supervision: bool = True
    model_type: str = "looped"


@dataclass
class HRMModelSettings:
    block_size: int = 512
    n_embd: int = 512
    n_head: int = 8
    n_fast_steps: int = 4
    n_slow_cycles: int = 3
    dropout: float = 0.0
    bias: bool = False
    rope_theta: float = 10000.0
    halt_epsilon: float = 1e-3
    ponder_penalty: float = 0.01
    one_step_gradient: bool = False
    model_type: str = "hierarchical"


@dataclass
class TrainingConstants:
    epochs: int = 5
    batch_size: int = 64
    lr: float = 1e-3
    n_train: int = 10000
    n_val: int = 1000
    mask_prompt: bool = True
    save_path: str = "math_gpt.pt"
    log_interval: int = 50
    seed: int = 42


@dataclass
class TestModelSettings:
    block_size: int = 16
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 32
    dropout: float = 0.0
    use_pos_emb: bool = True


@dataclass
class TestingConstants:
    model_settings: TestModelSettings = field(default_factory=TestModelSettings)
    seed: int = 42
    sample_equations: List[str] = field(
        default_factory=lambda: [
            "123+456=579",
            "10*5=50",
            "100-35=65",
            "12/4=3",
            "2^3=8",
        ]
    )
    edge_cases: List[str] = field(
        default_factory=lambda: ["0+0=0", "1-1=0", "0*99=0"]
    )
    invalid_expressions: List[str] = field(
        default_factory=lambda: ["12a+3b", "hello_world"]
    )
    benchmark_prompts: List[Dict[str, str]] = field(
        default_factory=lambda: [
            {"prompt": "12+15=", "expected": "27"},
            {"prompt": "48+52=", "expected": "100"},
            {"prompt": "120+80=", "expected": "200"},
            {"prompt": "99-33=", "expected": "66"},
            {"prompt": "50-25=", "expected": "25"},
            {"prompt": "6*7=", "expected": "42"},
            {"prompt": "9*9=", "expected": "81"},
            {"prompt": "15*4=", "expected": "60"},
            {"prompt": "2^3=", "expected": "8"},
            {"prompt": "3^3=", "expected": "27"},
        ]
    )


@dataclass
class T4HardwareConstants:
    gpu_name: str = "NVIDIA T4"
    gpu_count: int = 2
    vram_per_gpu_gb: float = 16.0
    compute_capability: Tuple[int, int] = (7, 5)
    precision: str = "fp16"
    use_amp: bool = True
    amp_dtype: str = "float16"
    cudnn_benchmark: bool = True
    batch_size_per_gpu: int = 64
    global_batch_size: int = 128
    ddp_backend: str = "nccl"
    pin_memory: bool = True
    grad_clip_norm: float = 1.0


def load_config(file_path: Optional[str] = None) -> Dict[str, Any]:
    target_path = file_path or CONFIG_FILE_PATH
    if os.path.exists(target_path):
        with open(target_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


_CONFIG_DATA = load_config()

TOKENIZER_CONSTANTS = TokenizerConstants(**_CONFIG_DATA.get("tokenizer", {}))
MODEL_CONSTANTS = ModelConstants(**_CONFIG_DATA.get("model", {}))
TRAINING_CONSTANTS = TrainingConstants(**_CONFIG_DATA.get("training", {}))

_test_dict = dict(_CONFIG_DATA.get("testing", {}))
_test_model_dict = _test_dict.pop("model_settings", {}) if "model_settings" in _test_dict else {}
TESTING_CONSTANTS = TestingConstants(
    model_settings=TestModelSettings(**_test_model_dict),
    **_test_dict,
)

T4_CONSTANTS = T4HardwareConstants(**_CONFIG_DATA.get("t4_hardware", {}))

CHINCHILLA_TOKEN_PARAM_RATIO: float = 20.0
NON_EMBEDDING_PARAMS_SUB20M: int = 18_915_328
CHINCHILLA_COMPUTE_OPTIMAL_TOKENS: int = int(NON_EMBEDDING_PARAMS_SUB20M * CHINCHILLA_TOKEN_PARAM_RATIO)


def get_device(local_rank: Optional[int] = None) -> Any:
    import torch
    if torch.cuda.is_available():
        if local_rank is not None:
            return torch.device(f"cuda:{local_rank}")
        if "LOCAL_RANK" in os.environ:
            return torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
        return torch.device("cuda:0")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def setup_t4_runtime(local_rank: Optional[int] = None) -> Any:
    import torch
    device = get_device(local_rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = T4_CONSTANTS.cudnn_benchmark
        if local_rank is not None:
            torch.cuda.set_device(local_rank)
        elif "LOCAL_RANK" in os.environ:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    return device
