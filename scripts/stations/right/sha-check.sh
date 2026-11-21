cd /models/Hy4-FP8
grep "safetensors$" ~/l6-4h/fp8-lfs.sha256 > ~/l6-4h/fp8-shards.sha256
echo "$(date) start $(wc -l < ~/l6-4h/fp8-shards.sha256) shards"
split -n l/16 ~/l6-4h/fp8-shards.sha256 /tmp/l64h-sha-
ls /tmp/l64h-sha-* | xargs -P 16 -I{} sh -c "nice sha256sum -c {} 2>&1" > ~/l6-4h/fp8-sha-results.txt
echo "OK $(grep -c ": OK$" ~/l6-4h/fp8-sha-results.txt) FAIL $(grep -vc ": OK$" ~/l6-4h/fp8-sha-results.txt)"
rm -f /tmp/l64h-sha-*
ls /models/Hy4-GGUF/*.gguf | xargs -P 4 -n 1 nice sha256sum > ~/l6-4h/gguf.sha256
cat ~/l6-4h/gguf.sha256; echo "$(date) SHA_DONE"
