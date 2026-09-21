#!/usr/bin/env python3
"""有界本机功能验证：启动独立服务、请求、保存结果、只清理本脚本启动的进程组。

用法：python3 run_smoke.py job.json（job 含 argv/env/port/framework/out）。
依赖：Python 标准库、容器自带推理框架。默认监听 loopback，避免对外开放服务。
回填：docs/details/evidence/memory/framework_runtime_validation_20260921.md。
"""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def request(url, body=None, timeout=30):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content = response.read().decode()
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            pass
        return {"status": response.status, "body": content}


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    out = Path(job["out"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise RuntimeError("Refusing to overwrite a completed run; choose another output directory")
    port = job["port"]
    with socket.socket() as sock:
        # 允许刚结束的本任务连接处于 TIME_WAIT；仍会拒绝正在监听的服务。
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))  # 已有服务绝不复用/停止。
    env = dict(os.environ, **job.get("env", {}))
    env.update(MSV_RUNTIME_CAPTURE="1", MSV_RUNTIME_OUT=str(out))
    env["PYTHONPATH"] = str(Path(__file__).parent) + os.pathsep + env.get("PYTHONPATH", "")
    result = {"job": job, "started": time.time(), "ready": False, "requests": [], "ok": False}
    (out / "job.json").write_text(json.dumps(job, indent=2) + "\n")
    with (out / "server.log").open("w") as log:
        proc = subprocess.Popen(job["argv"], env=env, stdout=log, stderr=log, start_new_session=True)
        result["server_pid"] = proc.pid
        (out / "server.pid").write_text(str(proc.pid) + "\n")
        try:
            deadline = time.monotonic() + job.get("startup_timeout", 600)
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    if request(f"http://127.0.0.1:{port}/health", timeout=2)["status"] == 200:
                        result["ready"] = True
                        break
                except Exception:
                    pass
                time.sleep(3)
            if not result["ready"]:
                raise RuntimeError(f"service not ready; process return code={proc.poll()}")
            for text in job.get("prompts", ["The capital of France is", "One plus one equals"]):
                if job["framework"] == "sglang":
                    path = "/generate"
                    body = {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": 24}}
                else:
                    path = "/v1/completions"
                    body = {"model": job["model"], "prompt": text, "temperature": 0, "max_tokens": 24}
                response = request(f"http://127.0.0.1:{port}{path}", body, timeout=180)
                result["requests"].append(response)
                payload = response["body"]
                output = payload.get("text", "") if job["framework"] == "sglang" else payload["choices"][0]["text"]
                if not isinstance(output, str) or not output.strip():
                    raise RuntimeError("empty generation; not a successful functional smoke")
            result["ok"] = True  # 非精度 benchmark；dummy 权重不判语义正确性。
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            result["returncode_before_cleanup"] = proc.poll()
            # 只对本任务创建的 session/process group 发信号，不用 pkill 框架名。
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            result["finished"] = time.time()
            result["server_returncode"] = proc.returncode
            (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
