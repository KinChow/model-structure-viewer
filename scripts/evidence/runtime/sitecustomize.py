"""只读 runtime 探针：在显式 MSV_RUNTIME_CAPTURE=1 时记录实际张量与 cache spec。

用法：PYTHONPATH 指向本目录，MSV_RUNTIME_OUT 指向本轮证据目录，然后正常启动框架。
依赖：运行容器自带 torch/vLLM/SGLang。不修改框架文件、分配参数或 forward 返回值。
回填：docs/details/evidence/memory/framework_runtime_validation_20260921.md。
"""
import dataclasses
import functools
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import time


def describe(value, depth=0):
    """只描述张量元数据；不复制 GPU 数据、不读取权重内容。"""
    import torch
    if isinstance(value, torch.Tensor):
        storage = value.untyped_storage()
        return {
            "shape": list(value.shape), "dtype": str(value.dtype),
            "device": str(value.device), "numel": value.numel(),
            "bytes": value.numel() * value.element_size(),
            "storage_ptr": storage.data_ptr(), "storage_bytes": storage.nbytes(),
            "offset": value.storage_offset(),
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if depth > 8:
        return {"class": type(value).__name__, "truncated": True}
    if isinstance(value, (list, tuple)):
        return [describe(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): describe(v, depth + 1) for k, v in value.items()}
    if dataclasses.is_dataclass(value):
        return {f.name: describe(getattr(value, f.name), depth + 1) for f in dataclasses.fields(value)}
    # 仅递归缓存对象；不会沿 model/engine 遍历任意对象图。
    module = type(value).__module__
    if "mem_cache" in module and hasattr(value, "__dict__"):
        return {k: describe(v, depth + 1) for k, v in vars(value).items()
                if not k.startswith("__") and k not in (
                    "memory_saver_adapter", "custom_mem_pool", "pool", "parent",
                )}
    return {"class": type(value).__name__}


def capture(obj, event):
    record = {"event": event, "pid": os.getpid(), "time": time.time(), "class": type(obj).__name__}
    for key in ("tp_rank", "tp_size", "is_draft_worker", "size", "page_size",
                "max_total_num_tokens", "num_mamba_layers"):
        value = getattr(obj, key, None)
        if value is not None:
            record[key] = describe(value)
    if event.endswith("load_model") and getattr(obj, "model", None) is not None:
        record["parameters"] = {name: describe(t) for name, t in obj.model.named_parameters()}
        record["model_class"] = type(obj.model).__name__
    else:
        record["allocation"] = describe(obj) if "mem_cache" in type(obj).__module__ else {
            k: describe(getattr(obj, k)) for k in (
                "kv_cache_config", "kv_caches", "token_to_kv_pool", "req_to_token_pool",
            ) if hasattr(obj, k)
        }
    out = Path(os.environ["MSV_RUNTIME_OUT"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"capture-{os.getpid()}.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def wrap(cls, method):
    original = getattr(cls, method, None)
    if original is None or getattr(original, "_msv_capture", False):
        return

    @functools.wraps(original)
    def observed(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            capture(self, f"{cls.__name__}.{method}")
        except Exception as exc:
            # 探针错误独立记录，不改变框架运行结果，也不伪装成已取到证据。
            out = Path(os.environ["MSV_RUNTIME_OUT"])
            out.mkdir(parents=True, exist_ok=True)
            with (out / f"capture-error-{os.getpid()}.txt").open("a") as handle:
                handle.write(f"{cls.__name__}.{method}: {type(exc).__name__}: {exc}\n")
        return result
    observed._msv_capture = True
    setattr(cls, method, observed)


TARGETS = {
    "sglang.srt.mem_cache.memory_pool": {
        "MambaPool": ["__init__"], "DSATokenToKVPool": ["__init__"],
        "MLATokenToKVPool": ["__init__"], "MHATokenToKVPool": ["__init__"],
    },
    "sglang.srt.mem_cache.deepseek_v4_memory_pool": {
        "DeepSeekV4TokenToKVPool": ["__init__"],
    },
    "sglang.srt.model_executor.model_runner": {
        "ModelRunner": ["load_model", "init_memory_pool"],
    },
    "vllm.v1.worker.gpu_model_runner": {
        "GPUModelRunner": ["load_model", "initialize_kv_cache"],
    },
    "vllm.v1.worker.gpu.model_runner": {
        "GPUModelRunner": ["load_model", "initialize_kv_cache"],
    },
}


class ObserverFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in TARGETS:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        loader = spec.loader
        original_exec = loader.exec_module

        def exec_observed(module):
            original_exec(module)
            for name, methods in TARGETS[fullname].items():
                cls = getattr(module, name, None)
                if cls is not None:
                    for method in methods:
                        wrap(cls, method)
        loader.exec_module = exec_observed
        return spec


if os.environ.get("MSV_RUNTIME_CAPTURE") == "1":
    sys.meta_path.insert(0, ObserverFinder())
