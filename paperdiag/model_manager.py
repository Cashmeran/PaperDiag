"""模型管理器：双后端自动检测（Ollama > llama.cpp portable > 报错）

后端优先级：
  1. Ollama (localhost:11434) — 如果用户已安装
  2. llama.cpp portable — 自动下载30MB便携包，解压即用

用法：
  paperdiag 会自动检测可用后端，无需用户配置。
  首次使用 llama.cpp 后端时，自动下载 ~30MB 便携包到缓存目录。

核显/iGPU 支持：
  llama.cpp CPU模式在所有x86-64 CPU上运行（含Intel/AMD核显笔记本）。
  4B Q4_K_M模型需要约6GB可用内存，8GB笔记本可运行。
  推理速度：核显本 ~5-10 tok/s，独显 ~50-200 tok/s。
"""

import json
import os
import sys
import time
import subprocess
import urllib.request
import urllib.error
import zipfile
import shutil
from pathlib import Path
from typing import Optional, Any

# ============================================================
#  配置
# ============================================================

CACHE_DIR = Path.home() / ".cache" / "paperdiag"
MODELS_DIR = CACHE_DIR / "models"
LLAMACPP_DIR = CACHE_DIR / "llamacpp"
LLAMACPP_BASE = "http://localhost:8080"

# llama.cpp Windows 预编译包（CPU版，avx2，最小体积）
LLAMACPP_CPU_URL = ""  # 由 setup 命令下载
LLAMACPP_CUDA_URL = ""  # 由 setup 命令下载
LLAMACPP_CPU_DIR = LLAMACPP_DIR / "cpu"
LLAMACPP_CUDA_DIR = LLAMACPP_DIR / "cuda"

# 模型配置
MODEL_CONFIGS = {
    "qwen3.5-4b": {
        "name": "Qwen3.5 4B Q4_K_XL",
        "filename": "Qwen3.5-4B-UD-Q4_K_XL.gguf",
        "context_length": 8192,
        "ram_required_gb": 6,
        "description": "C-Eval 85.1, IFEval 89.8 — 指令遵循王者",
    },
}
DEFAULT_MODEL = "qwen3.5-4b"


# ============================================================
#  后端检测与启动
# ============================================================

def _http_post(url: str, data: dict, timeout: int = 10) -> dict:
    """发送HTTP POST请求"""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_stream(url: str, data: dict, timeout: int = 120) -> str:
    """发送HTTP POST流式请求，拼接完整响应"""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    chunks = []
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            try:
                chunk = json.loads(line.decode("utf-8"))
                c = chunk.get("message", {}).get("content", "")
                if c:
                    chunks.append(c)
                if chunk.get("done"):
                    break
            except json.JSONDecodeError:
                continue
    return "".join(chunks)


def _has_nvidia_gpu() -> bool:
    """检测是否有 NVIDIA 显卡"""
    try:
        import subprocess
        r = subprocess.run(
            ['powershell', '-Command',
             'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name'],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return 'nvidia' in r.stdout.lower()
    except Exception:
        return False


def _check_llamacpp() -> bool:
    """检查llama.cpp server是否可用"""
    try:
        urllib.request.urlopen(f"{LLAMACPP_BASE}/v1/models", timeout=3)
        return True
    except Exception:
        return False


def _find_server_binary(base_dir: Path) -> Optional[Path]:
    """在目录中查找 llama-server 可执行文件"""
    exe_name = "llama-server.exe" if sys.platform == "win32" else "llama-server"
    direct = base_dir / exe_name
    if direct.exists():
        return direct
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f == exe_name or (f.startswith("llama-server") and f.endswith(".exe")):
                return Path(root) / f
    return None


def _start_llamacpp_server(model_path: str, use_cuda: bool = False) -> bool:
    """启动llama.cpp server（后台进程）"""
    if _check_llamacpp():
        return True

    engine_dir = LLAMACPP_CUDA_DIR if use_cuda else LLAMACPP_CPU_DIR
    binary = _find_server_binary(engine_dir)

    if not binary:
        raise RuntimeError(
            f"llama.cpp 未找到。请先运行: paperdiag setup\n"
            f"  或手动放置到: {engine_dir}"
        )

    engine_type = "CUDA" if use_cuda else "CPU"
    print(f"[model] Starting llama.cpp ({engine_type}) ...")

    try:
        kwargs = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        cmd = [str(binary), "-m", model_path, "--port", "8080",
               "-c", "8192", "-np", "4", "--no-webui",
               "--reasoning", "off"]

        # GPU加速：将所有层offload到GPU
        if use_cuda:
            cmd.extend(["-ngl", "99"])

        subprocess.Popen(cmd, **kwargs)

        for _ in range(30):
            time.sleep(1)
            if _check_llamacpp():
                print(f"[model] Server ready on :8080 ({engine_type})")
                return True

        raise RuntimeError("llama.cpp server 启动超时")
    except Exception as e:
        raise RuntimeError(f"llama.cpp 启动失败: {e}")


def setup_engines(proxy: Optional[str] = None) -> bool:
    """一次性下载 CPU + CUDA 两个 llama.cpp 引擎（安装时调用）

    下载到 ~/.cache/paperdiag/llamacpp/cpu/ 和 .../cuda/
    总计约 146MB。运行时不再下载。
    """
    import urllib.request as ureq

    # 设置代理
    if proxy:
        os.environ["HTTP_PROXY"] = proxy
        os.environ["HTTPS_PROXY"] = proxy

    # 获取最新 release
    api_url = "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=1"
    req = ureq.Request(api_url)
    req.add_header("User-Agent", "paperdiag")
    req.add_header("Accept", "application/vnd.github+json")

    try:
        data = json.loads(ureq.urlopen(req, timeout=15).read())
        assets = data[0]["assets"]
    except Exception as e:
        print(f"[setup] Failed to fetch release info: {e}")
        return False

    # 查找 CPU 和 CUDA 版本
    cpu_asset = None
    cuda_asset = None
    for a in assets:
        name = a["name"]
        if "cpu-x64" in name and name.endswith(".zip") and "arm" not in name:
            cpu_asset = a
        if "cuda-13.1" in name and name.endswith(".zip") and "cudart" not in name:
            cuda_asset = a

    if not cpu_asset:
        print("[setup] CPU binary not found in release")
        return False

    # 下载 CPU 版
    _download_and_extract(cpu_asset, LLAMACPP_CPU_DIR, "CPU")

    # 下载 CUDA 版（仅当有NVIDIA GPU时）
    if cuda_asset and _has_nvidia_gpu():
        _download_and_extract(cuda_asset, LLAMACPP_CUDA_DIR, "CUDA")

    print("[setup] Done. Engines ready.")
    return True


def _download_and_extract(asset: dict, dest_dir: Path, label: str):
    """下载并解压一个引擎"""
    import urllib.request as ureq

    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / "llamacpp.zip"

    # 已存在则跳过
    if _find_server_binary(dest_dir):
        print(f"[setup] {label} engine already cached")
        return

    size_mb = asset["size"] / 1024**2
    print(f"[setup] Downloading {label} engine ({size_mb:.0f}MB) ...")

    dl_req = ureq.Request(asset["browser_download_url"])
    dl_req.add_header("User-Agent", "paperdiag")
    with ureq.urlopen(dl_req, timeout=180) as resp:
        with open(zip_path, "wb") as f:
            f.write(resp.read())

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    os.remove(zip_path)

    binary = _find_server_binary(dest_dir)
    if binary:
        print(f"[setup] {label} engine ready: {binary.name}")


# ============================================================
#  统一接口
# ============================================================

class LLMBackend:
    """统一的LLM后端接口，自动检测可用后端"""

    def __init__(self, model_key: str = DEFAULT_MODEL):
        self.model_key = model_key
        self.config = MODEL_CONFIGS[model_key]
        self._backend: Optional[str] = None  # "ollama" or "llamacpp"

    def ensure_ready(self):
        """确保后端可用，自动检测GPU并选择最优引擎"""
        if self._backend:
            return

        model_path = MODELS_DIR / self.config["filename"]
        if not model_path.exists():
            raise RuntimeError(
                f"模型文件不存在: {model_path}\n"
                f"请将 {self.config['filename']} 放到 {MODELS_DIR}"
            )

        # GPU检测 → 二选一
        use_cuda = _has_nvidia_gpu()
        backend_name = "llamacpp-cuda" if use_cuda else "llamacpp-cpu"
        print(f"[model] GPU: {'NVIDIA detected → CUDA' if use_cuda else 'not found → CPU'}")

        _start_llamacpp_server(str(model_path), use_cuda=use_cuda)
        self._backend = backend_name

    def chat(self, messages: list[dict],
             temperature: float = 0.3,
             max_tokens: int = 1024) -> dict[str, Any]:
        """统一聊天接口"""
        self.ensure_ready()

        data = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        result = _http_post(
            f"{LLAMACPP_BASE}/v1/chat/completions", data, timeout=180
        )
        raw = result["choices"][0]["message"]["content"]
        return {"content": _strip_thinking(raw)}


def _strip_thinking(text: str) -> str:
    """去除Qwen3.5思考标签（包括未闭合的）"""
    import re
    # 去除闭合的 <think>...</think>
    text = re.sub(r'<think>[\s\S]*?</think>', '', text)
    # 去除未闭合的 <think>（从<think>到文本末尾）
    text = re.sub(r'<think>[\s\S]*$', '', text)
    # 去除空标签
    text = re.sub(r'<think\s*/>', '', text)
    return text.strip()


# 全局单例
_backend: Optional[LLMBackend] = None


def get_backend(model_key: str = DEFAULT_MODEL) -> LLMBackend:
    global _backend
    if _backend is None:
        _backend = LLMBackend(model_key)
    return _backend


# ============================================================
#  模块级接口（供 llm_diagnose / llm_rewriter 调用）
# ============================================================

def chat(messages: list[dict],
         model_key: str = DEFAULT_MODEL,
         temperature: float = 0.3,
         max_tokens: int = 1024,
         **kwargs) -> dict[str, Any]:
    return get_backend(model_key).chat(
        messages, temperature=temperature, max_tokens=max_tokens
    )


def ensure_ollama_ready(model_key: str = DEFAULT_MODEL):
    """兼容旧接口"""
    get_backend(model_key).ensure_ready()


def is_model_available(model_key: str = DEFAULT_MODEL) -> bool:
    config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS[DEFAULT_MODEL])
    return (MODELS_DIR / config["filename"]).exists()


def is_backend_available() -> bool:
    return _check_llamacpp()


def get_model_info(model_key: str = DEFAULT_MODEL) -> dict:
    config = MODEL_CONFIGS.get(model_key, MODEL_CONFIGS[DEFAULT_MODEL])
    model_path = MODELS_DIR / config["filename"]
    return {
        "key": model_key,
        "name": config["name"],
        "model_cached": model_path.exists(),
        "model_size_gb": round(model_path.stat().st_size / (1024**3), 1) if model_path.exists() else None,
        "ollama_running": _check_ollama(),
        "llamacpp_running": _check_llamacpp(),
        "backend_available": is_backend_available(),
    }


def list_available_models() -> list[dict]:
    return [
        {
            "key": k,
            "name": c["name"],
            "cached": (MODELS_DIR / c["filename"]).exists(),
            "description": c["description"],
        }
        for k, c in MODEL_CONFIGS.items()
    ]
