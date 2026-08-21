"""
Multi-threaded chunked downloader for the ONNX reranker model.
ModelScope CDN supports Range requests -> 8 parallel connections.

Usage: python download_model.py [url] [output_path]
"""
import os
import sys
import time
import threading
import urllib.request

URL = "https://modelscope.cn/models/onnx-community/bge-reranker-v2-m3-ONNX/resolve/master/onnx/model_quantized.onnx"
OUT = "./models/bge-reranker-v2-m3-onnx/onnx/model_quantized.onnx"
NUM_THREADS = 8

progress = {"done": 0, "lock": threading.Lock()}
start = time.time()


def get_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=15) as r:
        return int(r.headers.get("Content-Length", 0))


def download_chunk(url: str, out: str, start_byte: int, end_byte: int, idx: int):
    headers = {"Range": f"bytes={start_byte}-{end_byte}"}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r, open(out, "r+b") as f:
        f.seek(start_byte)
        while True:
            chunk = r.read(1024 * 256)
            if not chunk:
                break
            f.write(chunk)
            with progress["lock"]:
                progress["done"] += len(chunk)
            pct = progress["done"] / max(1, end_byte + 1) * 100
            elapsed = time.time() - start
            speed = progress["done"] / max(1, elapsed) / 1024
            print(
                f"\r{idx}: {progress['done']/1024/1024:.1f}MB ({pct:.1f}%) "
                f"{speed:.0f}KB/s",
                end="", flush=True,
            )


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else URL
    out = sys.argv[2] if len(sys.argv) > 2 else OUT
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)

    total = get_size(url)
    print(f"Total size: {total/1024/1024:.1f}MB, threads: {NUM_THREADS}")

    with open(out, "wb") as f:
        f.truncate(total)

    part = total // NUM_THREADS
    threads = []
    for i in range(NUM_THREADS):
        s = i * part
        e = (i + 1) * part - 1 if i < NUM_THREADS - 1 else total - 1
        t = threading.Thread(target=download_chunk, args=(url, out, s, e, i), daemon=True)
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    elapsed = time.time() - start
    print(f"\nDONE: {out} in {elapsed:.1f}s "
          f"({total/1024/1024/(elapsed/60):.1f}MB/min)")


if __name__ == "__main__":
    main()
