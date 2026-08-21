#!/bin/bash
# Parallel chunked download via curl (8 processes) with progress + resume.
URL="https://modelscope.cn/models/onnx-community/bge-reranker-v2-m3-ONNX/resolve/master/onnx/model_quantized.onnx"
OUT="/Users/developer/Project/rag-server/models/bge-reranker-v2-m3-onnx/onnx/model_quantized.onnx"
TOTAL=570727094
N=8
TMP="/Users/developer/Project/rag-server/models/bge-reranker-v2-m3-onnx/onnx/.parts"
mkdir -p "$TMP"

# Pre-allocate file
if [ ! -f "$OUT" ] || [ "$(stat -f%z "$OUT")" != "$TOTAL" ]; then
  /usr/bin/truncate -s $TOTAL "$OUT" 2>/dev/null || dd if=/dev/zero of="$OUT" bs=1 count=0 seek=$TOTAL 2>/dev/null
fi

PART=$((TOTAL / N))
dl_chunk() {
  local i=$1
  local s=$((i * PART))
  local e
  if [ $i -eq $((N - 1)) ]; then e=$((TOTAL - 1)); else e=$(((i + 1) * PART - 1)); fi
  local marker="$TMP/$i.ok"
  if [ -f "$marker" ]; then echo "chunk $i already done"; return; fi
  curl -sL --max-time 3600 -r ${s}-${e} -o "$TMP/$i.bin" "$URL"
  if [ $? -eq 0 ] && [ "$(stat -f%z "$TMP/$i.bin")" -eq $((e - s + 1)) ]; then
    dd if="$TMP/$i.bin" of="$OUT" bs=1048576 seek=$((s / 1048576)) conv=notrunc 2>/dev/null
    rm -f "$TMP/$i.bin"
    touch "$marker"
    echo "chunk $i done ($s-$e)"
  else
    echo "chunk $i FAILED"
    rm -f "$TMP/$i.bin"
  fi
}

export -f dl_chunk
export TMP OUT URL TOTAL PART N
seq 0 $((N - 1)) | xargs -P $N -I{} bash -c 'dl_chunk {}'

# Verify
SIZE=$(stat -f%z "$OUT")
echo "Final size: $SIZE / $TOTAL"
DONE=$(ls "$TMP" | grep -c "\.ok$" 2>/dev/null || echo 0)
echo "Chunks done: $DONE/$N"
