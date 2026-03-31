# Thin convenience wrappers for the existing repo scripts.
# The shell scripts remain the source of truth for setup and worker behavior.

ENV_MANAGER ?= auto

.DEFAULT_GOAL := help

.PHONY: help bootstrap verify worker web

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make bootstrap           Wraps ./scripts/setup-worker.sh --env-manager $(ENV_MANAGER)' \
		'  make verify              Wraps ./scripts/verify-worker.sh' \
		'  make worker              Wraps ./scripts/start-worker.sh --env-manager $(ENV_MANAGER)' \
		'  make web                 Wraps cd web && npm run dev' \
		'' \
		'Override the worker env manager when needed, for example:' \
		'  make bootstrap ENV_MANAGER=conda' \
		'  make worker ENV_MANAGER=micromamba'

bootstrap:
	./scripts/setup-worker.sh --env-manager "$(ENV_MANAGER)"

verify:
	./scripts/verify-worker.sh

worker:
	./scripts/start-worker.sh --env-manager "$(ENV_MANAGER)"

web:
	cd web && npm run dev
