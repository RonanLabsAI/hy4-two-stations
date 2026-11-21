for f in mxfp4 mxfp8; do for lay in 1 20 40 60 77; do
echo "== $f layer $lay $(date +%T)"
docker run --rm --name l64h-b12x --device nvidia.com/gpu=GPU-2fd00932-6173-73f2-9bf0-33854c69b3ac -v /models/Hy4-FP8:/model:ro -v $HOME/hy4-e3/repo:/e3:ro -v $HOME/dsv4pro-e3/b12x:/b12x:ro -e PYTHONPATH=/b12x -e E3_HOOK_DIR=/e3/hook -e E3_CONFIG=/model/config.json --entrypoint python3 vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b /e3/tests/b12x_layer_check_hy4.py --model /model --layer $lay --experts 0-35 --wfmt $f 2>&1 | grep -E "cos\(|gate:|Error|error" ; echo "rc=${PIPESTATUS[0]}"
done; done
echo "== control: swapped w13 layout must FAIL"
docker run --rm --name l64h-b12x --device nvidia.com/gpu=GPU-2fd00932-6173-73f2-9bf0-33854c69b3ac -v /models/Hy4-FP8:/model:ro -v $HOME/hy4-e3/repo:/e3:ro -v $HOME/dsv4pro-e3/b12x:/b12x:ro -e PYTHONPATH=/b12x -e E3_HOOK_DIR=/e3/hook -e E3_CONFIG=/model/config.json --entrypoint python3 vllm/vllm-openai:nightly-af7f9488c2210d67e1033ecdc845b087ee7fe92b /e3/tests/b12x_layer_check_hy4.py --model /model --layer 20 --experts 0-35 --wfmt mxfp4 --layout w13 2>&1 | grep -E "cos\(|gate:"; echo "rc=${PIPESTATUS[0]}"
echo B12X_DONE
