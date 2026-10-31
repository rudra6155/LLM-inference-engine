PYTHON ?= python3

.PHONY: install install-baselines test benchmark profile roofline tune optimized clean

# Editable install is required: the C++ sources live in src/csrc, next to (not
# inside) the `engine` package, and are located relative to the source tree.
install:
	$(PYTHON) -m pip install -e ".[awq,bench,dev]"
	$(PYTHON) -c "from engine.cpp_ops import load_custom_ops; load_custom_ops(verbose=True)"

# Baselines get their own virtualenvs: vLLM 0.32.0 needs transformers>=5.16.1,
# which conflicts with the autoawq-era pin in pyproject.toml. llama-cpp-python
# is built from source with CUDA, so `nvcc` must be on PATH.
install-baselines:
	$(PYTHON) -m venv .venvs/vllm
	.venvs/vllm/bin/python -m pip install "vllm==0.32.0" nvidia-ml-py
	$(PYTHON) -m venv .venvs/llamacpp
	CMAKE_ARGS="-DGGML_CUDA=on" .venvs/llamacpp/bin/python -m pip install "llama-cpp-python==0.3.36" numpy nvidia-ml-py

test:
	$(PYTHON) -m pytest tests

benchmark:
	bash scripts/benchmark.sh

profile:
	bash scripts/profile.sh

roofline:
	$(PYTHON) scripts/roofline.py

# Needs the GPU: sweeps tile / warps / stages / split-K for the decode kernels and writes
# results/tuning/best_configs.json (use it with EngineOptions(tuned_config=...) or BACKENDS="engine-tuned").
tune:
	$(PYTHON) scripts/tune_kernels.py

# Tests, tuning, and the eager / CUDA-graph / tuned benchmark against llama.cpp, into results/run_c/.
optimized:
	bash scripts/wsl_run_optimized.sh

clean:
	rm -rf build .pytest_cache src/*.egg-info src/engine/__pycache__ tests/__pycache__
