SHELL  := /bin/bash
ARGS   ?=

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show available targets
	@printf "\nUsage: make \033[36m<target>\033[0m [ARGS=\"...\"]\n\n"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'
	@printf "\nExamples:\n"
	@printf "  make train ARGS=\"training=lora\"\n"
	@printf "  make sweep ARGS=\"training.learning_rate=1e-4,3e-4,1e-3\"\n\n"

.PHONY: build
build: ## Build Docker image (auto-detects GPU platform)
	@./scripts/build.sh

.PHONY: shell
shell: ## Launch interactive container shell
	@./scripts/shell.sh

.PHONY: data
data: ## Prepare dataset (download, split, store metadata)
	@./scripts/cli.sh stage=data $(ARGS)

.PHONY: train
train: ## Run training with NeMo/Megatron
	@./scripts/cli.sh stage=train $(ARGS)

.PHONY: fetch
fetch: ## Fetch output.zip from CUDA/ROCM servers
	@./scripts/download.sh

.PHONY: eval
eval: ## Evaluate: load benchmarks, build compare.png
	@python scripts/eval.py $(ARGS)

.PHONY: wrap
wrap: ## Package outputs into output.zip
	@./scripts/cli.sh stage=wrap $(ARGS)

.PHONY: purge
purge: ## Remove outputs and caches
	@./scripts/cli.sh stage=purge $(ARGS)

.PHONY: purge-all
purge-all: ## Remove outputs, caches, and data
	@./scripts/cli.sh stage=purge with_data=true $(ARGS)

.PHONY: all
all: ## Full pipeline: data -> train -> wrap
	@./scripts/cli.sh stage=all $(ARGS)

.PHONY: sweep
sweep: ## Hydra multirun sweep (pass grid via ARGS)
	@./scripts/cli.sh --multirun $(ARGS)
