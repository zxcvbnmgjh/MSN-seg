import os
import sys
import time
import json
import signal
import argparse
import subprocess
import traceback


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_parent():
    import torch

    print_header("Environment Summary")
    print(f"Python               : {sys.version.split()[0]}")
    print(f"PyTorch              : {torch.__version__}")
    print(f"PyTorch CUDA build   : {torch.version.cuda}")
    print(f"CUDA available       : {torch.cuda.is_available()}")
    print(f"cuDNN available      : {torch.backends.cudnn.is_available()}")
    print(f"cuDNN version        : {torch.backends.cudnn.version()}")
    print(f"cudnn.enabled        : {torch.backends.cudnn.enabled}")
    print(f"cudnn.benchmark      : {torch.backends.cudnn.benchmark}")
    print(f"cudnn.deterministic  : {torch.backends.cudnn.deterministic}")

    if torch.cuda.is_available():
        print(f"GPU count            : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU[{i}] name         : {torch.cuda.get_device_name(i)}")
        prop = torch.cuda.get_device_properties(0)
        print(f"GPU[0] total memory  : {prop.total_memory / 1024**3:.2f} GB")
    else:
        print("No CUDA device found. This script is for CUDA/cuDNN diagnosis.")
        return

    print_header("Subprocess Tests")
    cases = [
        {"name": "cudnn_off", "cudnn": 0, "benchmark": 0, "deterministic": 0, "amp": 0},
        {"name": "cudnn_on_safe", "cudnn": 1, "benchmark": 0, "deterministic": 1, "amp": 0},
        {"name": "cudnn_on_default", "cudnn": 1, "benchmark": 0, "deterministic": 0, "amp": 0},
        {"name": "cudnn_on_benchmark", "cudnn": 1, "benchmark": 1, "deterministic": 0, "amp": 0},
    ]

    results = []
    for case in cases:
        cmd = [
            sys.executable,
            __file__,
            "--child",
            "--name", case["name"],
            "--cudnn", str(case["cudnn"]),
            "--benchmark", str(case["benchmark"]),
            "--deterministic", str(case["deterministic"]),
            "--amp", str(case["amp"]),
        ]
        print(f"\n[RUN] {case['name']}")
        proc = subprocess.run(cmd, text=True, capture_output=True)
        result = {
            "name": case["name"],
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        results.append(result)

        if proc.returncode == 0:
            print(f"  -> PASS")
        elif proc.returncode < 0:
            print(f"  -> FAIL by signal {-proc.returncode}")
        else:
            print(f"  -> FAIL code {proc.returncode}")

    print_header("Detailed Results")
    for r in results:
        print(f"\n--- {r['name']} | returncode={r['returncode']} ---")
        if r["stdout"].strip():
            print("[stdout]")
            print(r["stdout"].strip())
        if r["stderr"].strip():
            print("[stderr]")
            print(r["stderr"].strip())

    print_header("Quick Interpretation")
    passed = [r["name"] for r in results if r["returncode"] == 0]
    failed = [r["name"] for r in results if r["returncode"] != 0]
    print("PASS:", passed if passed else "None")
    print("FAIL:", failed if failed else "None")

    if "cudnn_off" in passed and all(name != "cudnn_off" for name in passed[1:]) or (
        "cudnn_off" in passed and len(passed) == 1
    ):
        print(
            "\nLikely conclusion: the model and GPU path itself work, "
            "but the cuDNN convolution path on this machine is unstable for this workload."
        )
    elif len(passed) == len(results):
        print("\nAll tested modes passed. The previous crash may depend on training code, data, or AMP.")
    else:
        print("\nMixed result. Compare which cuDNN flags correlate with crashes.")


def build_model():
    """
    Try to import your actual model first.
    Fallback to a toy model that includes conv / depthwise conv / interpolate.
    """
    try:
        from src.FusionSegNet import FusionSegNet
        model = FusionSegNet(in_channels=3, base_filters=64, num_classes=1)
        model_name = "FusionSegNet"
        return model, model_name
    except Exception as e:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class ToyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 64, kernel_size=1)
                self.dw = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
                self.pw = nn.Conv2d(64, 128, kernel_size=1)
                self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
                self.out = nn.Conv2d(64, 1, kernel_size=1)

            def forward(self, x):
                x = self.conv1(x)
                x = F.gelu(x)
                x = self.dw(x)
                x = self.pw(x)
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                x = self.conv2(x)
                x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
                x = self.out(x)
                return x

        print(f"[WARN] Could not import real model, fallback to ToyNet. Import error: {repr(e)}")
        return ToyNet(), "ToyNet"


def run_child(args):
    import torch

    # Optional: make crash location easier to correlate
    os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

    torch.backends.cudnn.enabled = bool(args.cudnn)
    torch.backends.cudnn.benchmark = bool(args.benchmark)
    torch.backends.cudnn.deterministic = bool(args.deterministic)

    device = torch.device("cuda:0")
    model, model_name = build_model()
    model = model.to(device).train()

    print(f"name                : {args.name}")
    print(f"model               : {model_name}")
    print(f"device              : {device}")
    print(f"cudnn.enabled       : {torch.backends.cudnn.enabled}")
    print(f"cudnn.benchmark     : {torch.backends.cudnn.benchmark}")
    print(f"cudnn.deterministic : {torch.backends.cudnn.deterministic}")
    print(f"AMP enabled         : {bool(args.amp)}")

    x = torch.randn(1, 3, 512, 512, device=device, dtype=torch.float32).contiguous()
    print(f"input.shape         : {tuple(x.shape)}")
    print(f"input.dtype         : {x.dtype}")
    print(f"input.is_contiguous : {x.is_contiguous()}")

    start = time.time()
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=bool(args.amp)):
            y = model(x)
            loss = y.mean()

        print(f"output.shape        : {tuple(y.shape)}")
        print(f"loss                : {float(loss.detach().cpu()) :.6f}")

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.time() - start
        mem = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"max_mem_allocated   : {mem:.2f} MB")
        print(f"elapsed             : {elapsed:.3f} s")
        print("STATUS              : PASS")
    except Exception as e:
        print("STATUS              : PYTHON_EXCEPTION")
        print(repr(e))
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--name", type=str, default="case")
    parser.add_argument("--cudnn", type=int, default=1)
    parser.add_argument("--benchmark", type=int, default=0)
    parser.add_argument("--deterministic", type=int, default=0)
    parser.add_argument("--amp", type=int, default=0)
    args = parser.parse_args()

    if args.child:
        run_child(args)
    else:
        run_parent()