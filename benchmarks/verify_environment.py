import os
import sys
import time
import torch
import torch.nn as nn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))
sys.path.insert(0, ROOT_DIR)

from src.model import MathGPT, GPTConfig
from src.constants import T4_CONSTANTS


def _run_cpu_smoke_test():
    print("\nRunning local CPU smoke test...")
    cfg = GPTConfig(
        vocab_size=27,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        dropout=0.0,
        pos_emb_type="rope",
    )
    model = MathGPT(cfg)
    x = torch.randint(0, 27, (2, 32), dtype=torch.long)
    y = torch.randint(0, 27, (2, 32), dtype=torch.long)
    logits, loss = model(x, y)
    print(f"  [OK] Model Forward Pass: Logits shape={logits.shape}")
    loss.backward()
    print(f"  [OK] Model Backward Pass: Loss={loss.item():.4f}")
    print("  [OK] CPU Smoke Test Completed.")


def benchmark_fp16_gemm(device: torch.device, matrix_dim: int = 4096, warmup: int = 5, reps: int = 20):
    print(f"\nBenchmarking FP16 Tensor Core GEMM on {device} ({matrix_dim}x{matrix_dim})...")
    a = torch.randn(matrix_dim, matrix_dim, device=device, dtype=torch.float16)
    b = torch.randn(matrix_dim, matrix_dim, device=device, dtype=torch.float16)

    for _ in range(warmup):
        torch.mm(a, b)
    torch.cuda.synchronize(device)

    start = time.time()
    for _ in range(reps):
        torch.mm(a, b)
    torch.cuda.synchronize(device)
    elapsed = (time.time() - start) / reps

    flops = 2 * (matrix_dim ** 3)
    tflops = (flops / elapsed) / 1e12
    print(f"  [OK] FP16 GEMM: {elapsed * 1000:.2f} ms | Performance: {tflops:.2f} TFLOPS")


def run_diagnostics():
    print("=" * 75)
    print("NVIDIA T4 DUAL CLOUD RUNTIME DIAGNOSTIC SUITE")
    print("=" * 75)
    print(f"Python Version:  {sys.version.split()[0]}")
    print(f"PyTorch Version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:  {cuda_available}")

    if not cuda_available:
        print("\nNotice: CUDA is not active in this session. Running CPU smoke test.")
        _run_cpu_smoke_test()
        return

    print(f"CUDA Version:    {torch.version.cuda}")
    cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
    print(f"cuDNN Version:   {cudnn_version}")

    gpu_count = torch.cuda.device_count()
    print(f"GPU Count:       {gpu_count} (Expected: 2 for Dual T4)")

    for i in range(gpu_count):
        prop = torch.cuda.get_device_properties(i)
        vram_gb = prop.total_memory / (1024**3)
        cap = f"{prop.major}.{prop.minor}"
        print(f"GPU [{i}]: {prop.name} | VRAM: {vram_gb:.2f} GB | Compute: {cap} | SMs: {prop.multi_processor_count}")

    for i in range(gpu_count):
        benchmark_fp16_gemm(torch.device(f"cuda:{i}"))

    print("\nRunning end-to-end MathGPT FP16 AMP training verification...")
    dev0 = torch.device("cuda:0")
    cfg = GPTConfig(
        vocab_size=27,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
        pos_emb_type="rope",
    )
    model = MathGPT(cfg).to(dev0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    x = torch.randint(0, 27, (8, 64), device=dev0, dtype=torch.long)
    y = torch.randint(0, 27, (8, 64), device=dev0, dtype=torch.long)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        _, loss = model(x, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    print(f"  [OK] Training Step Completed: Loss={loss.item():.4f}")
    print("=" * 75)
    print("ALL ENVIRONMENT DIAGNOSTICS PASSED")
    print("=" * 75)


if __name__ == "__main__":
    run_diagnostics()
