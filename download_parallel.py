#!/usr/bin/env python
"""
Byte-precise parallel downloader for the ONNX reranker model.
- Downloads N chunks concurrently via curl (goes through the system proxy).
- Merges chunks into the final file with byte-exact seek (Python file ops,
  NOT dd block seek — dd's block-aligned seek corrupts non-aligned offsets).
- Verifies final file size.
"""
import os
import subprocess
import sys
import concurrent.futures

URL = "https://modelscope.cn/models/onnx-community/bge-reranker-v2-m3-ONNX/resolve/master/onnx/model_quantized.onnx"
TOTAL = 570727094  # verified via Content-Range earlier
OUT = "/Users/developer/Project/rag-server/models/bge-reranker-v2-m3-onnx/onnx/model_quantized.onnx"
PARTS = "/Users/developer/Project/rag-server/models/bge-reranker-v2-m3-onnx/onnx/.parts"
N = 8


def download_chunk(i: int, start: int, end: int) -> bool:
    part_file = os.path.join(PARTS, f"{i}.bin")
    if os.path.exists(part_file) and os.path.getsize(part_file) == (end - start + 1):
        print(f"chunk {i}: already complete, skip", flush=True)
        return True
    cmd = ["curl", "-sL", "--max-time", "1800", "-r", f"{start}-{end}", "-o", part_file, URL]
    r = subprocess.run(cmd, capture_output=True)
    ok = r.returncode == 0 and os.path.exists(part_file) and os.path.getsize(part_file) == (end - start + 1)
    if ok:
        print(f"chunk {i}: OK ({start}-{end})", flush=True)
    else:
        print(f"chunk {i}: FAILED rc={r.returncode}", flush=True)
        if os.path.exists(part_file):
            os.remove(part_file)
    return ok


def main():
    os.makedirs(PARTS, exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)  # fresh, avoids stale bytes at wrong offsets

    # Pre-allocate exact size
    with open(OUT, "wb") as f:
        f.truncate(TOTAL)

    part = TOTAL // N
    ranges = [(i, i * part, ((i + 1) * part - 1 if i < N - 1 else TOTAL - 1)) for i in range(N)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
        futures = {ex.submit(download_chunk, i, s, e): i for i, s, e in ranges}
        for fut in concurrent.futures.as_completed(futures):
            if not fut.result():
                sys.exit(f"chunk {futures[fut]} failed — rerun to resume (complete chunks are skipped)")

    # Byte-exact merge
    for i, start, end in ranges:
        part_file = os.path.join(PARTS, f"{i}.bin")
        with open(part_file, "rb") as src, open(OUT, "r+b") as dst:
            dst.seek(start)
            dst.write(src.read())
        os.remove(part_file)
        print(f"merged chunk {i}", flush=True)

    size = os.path.getsize(OUT)
    print(f"FINAL SIZE: {size} / expected {TOTAL} -> {'OK' if size == TOTAL else 'MISMATCH'}")


if __name__ == "__main__":
    main()
