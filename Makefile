# Snakebot Pipeline v2 — Top-level Build
.PHONY: all engine test-engine test-env train export submit clean

all: engine

# Build C++ engine shared library
engine:
	$(MAKE) -C engine all

# Run C++ engine tests
test-engine:
	$(MAKE) -C engine test

# Build engine + run Python environment tests
test-env: engine
	cd $(CURDIR) && python3 test_env.py

# Run all tests
test: test-engine test-env

# Train (default: 1M steps)
train: engine
	python3 train.py --total-steps 1000000

# Quick training smoke test (10K steps)
train-smoke: engine
	python3 train.py --total-steps 10000 --warmup-steps 5000 --n-eval-games 10 --eval-freq 5000

# Export weights
export:
	python3 export_weights.py

# Generate submission
submit: export
	python3 generate_submission.py
	g++ -std=c++17 -O2 -o submission submission.cpp
	@echo "Submission compiled: submission"

clean:
	$(MAKE) -C engine clean
	rm -f submission submission.cpp
	rm -rf checkpoints/ logs/ __pycache__/
