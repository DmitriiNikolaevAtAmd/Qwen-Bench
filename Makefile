# Qwen-Bench pipeline orchestration
# Usage: make <target> [ARGS="hydra overrides"]
#
# Examples:
#   make data
#   make train
#   make train ARGS="training=full"
#   make train ARGS="training.learning_rate=1e-3"
#   make sweep ARGS="training.learning_rate=1e-4,3e-4,1e-3"
#   make all

SHELL  := /bin/bash
ARGS   ?=

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@printf "\nUsage: make \033[36m<target>\033[0m [ARGS=\"...\"]\n\n"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@printf "\nExamples:\n"
	@printf "  make train ARGS=\"training=full\"\n"
	@printf "  make sweep ARGS=\"training.learning_rate=1e-4,3e-4,1e-3\"\n\n"

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

.PHONY: build
build: ## Build Docker image (auto-detects GPU platform)
	@./scripts/build.sh

.PHONY: shell
shell: ## Launch interactive container shell
	@./scripts/entry.sh

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

.PHONY: data
data: ## Prepare dataset (download, split, store metadata)
	@./scripts/run.sh stage=data $(ARGS)

.PHONY: train
train: ## Run training with LLaMA Factory
	@./scripts/run.sh stage=train $(ARGS)

.PHONY: wrap
wrap: ## Package outputs into output.zip
	@./scripts/run.sh stage=wrap $(ARGS)

.PHONY: purge
purge: ## Remove outputs and caches
	@./scripts/run.sh stage=purge $(ARGS)

.PHONY: all
all: ## Full pipeline: data -> train -> wrap
	@./scripts/run.sh stage=all $(ARGS)

# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

.PHONY: sweep
sweep: ## Hydra multirun sweep (pass grid via ARGS)
	@./scripts/run.sh --multirun $(ARGS)
