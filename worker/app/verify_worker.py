from __future__ import annotations

import argparse
import json
import subprocess
import sys


PROBE_DEFINITIONS = {
    "shared": {
        "label": "shared worker env",
        "code": "import cv2, fastapi, torch, diffsynth, modelscope, sam2, sam3; import app.main, app.runners.effecterase_remove",
    },
    "sam": {
        "label": "SAM env",
        "code": "import fastapi, torch, sam2, sam3; import app.main",
    },
    "remove": {
        "label": "EffectErase env",
        "code": "import cv2, torch, diffsynth, modelscope; import app.runners.effecterase_remove",
    },
}


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _probe_env(role: str, env_name: str) -> int:
    probe = PROBE_DEFINITIONS[role]
    report = {
        "envName": env_name,
        "role": role,
        "label": probe["label"],
        "imports": {
            "ok": False,
            "probe": probe["code"],
            "error": None,
        },
        "cuda": {
            "ok": False,
            "torchImportOk": False,
            "torchVersion": None,
            "cudaAvailable": False,
            "deviceCount": 0,
            "firstDeviceName": None,
            "error": None,
        },
    }

    try:
        exec(probe["code"], {})
        report["imports"]["ok"] = True
    except Exception as error:
        report["imports"]["error"] = _error_text(error)

    try:
        import torch
    except Exception as error:
        report["cuda"]["error"] = _error_text(error)
        print(json.dumps(report))
        return 0

    report["cuda"]["torchImportOk"] = True
    report["cuda"]["torchVersion"] = getattr(torch, "__version__", None)

    try:
        report["cuda"]["cudaAvailable"] = bool(torch.cuda.is_available())
        report["cuda"]["deviceCount"] = int(torch.cuda.device_count())
        if report["cuda"]["deviceCount"] > 0:
            report["cuda"]["firstDeviceName"] = torch.cuda.get_device_name(0)
        report["cuda"]["ok"] = report["cuda"]["cudaAvailable"] and report["cuda"]["deviceCount"] > 0
    except Exception as error:
        report["cuda"]["error"] = _error_text(error)

    print(json.dumps(report))
    return 0


def _manager_command(manager: str, env_name: str, *command: str) -> list[str]:
    if manager == "conda":
        return ["conda", "run", "--no-capture-output", "-n", env_name, *command]
    return ["micromamba", "run", "-n", env_name, *command]


def _run_probe(manager: str, worker_env: str, env_name: str, role: str) -> dict:
    if env_name == worker_env:
        command = [sys.executable, "-m", "app.verify_worker", "probe-env", "--role", role, "--env-name", env_name]
    else:
        # The aggregate command runs inside the worker env, so only cross-env
        # checks need another manager invocation.
        command = _manager_command(manager, env_name, "python", "-m", "app.verify_worker", "probe-env", "--role", role, "--env-name", env_name)

    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        error_text = completed.stderr.strip() or completed.stdout.strip() or "Probe command exited non-zero."
        return {
            "envName": env_name,
            "role": role,
            "label": PROBE_DEFINITIONS[role]["label"],
            "imports": {
                "ok": False,
                "probe": PROBE_DEFINITIONS[role]["code"],
                "error": error_text,
            },
            "cuda": {
                "ok": False,
                "torchImportOk": False,
                "torchVersion": None,
                "cudaAvailable": False,
                "deviceCount": 0,
                "firstDeviceName": None,
                "error": error_text,
            },
        }

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        output = completed.stdout.strip() or completed.stderr.strip() or "Probe command did not emit JSON."
        return {
            "envName": env_name,
            "role": role,
            "label": PROBE_DEFINITIONS[role]["label"],
            "imports": {
                "ok": False,
                "probe": PROBE_DEFINITIONS[role]["code"],
                "error": output,
            },
            "cuda": {
                "ok": False,
                "torchImportOk": False,
                "torchVersion": None,
                "cudaAvailable": False,
                "deviceCount": 0,
                "firstDeviceName": None,
                "error": output,
            },
        }


def _model_report() -> dict:
    from app.core.config import get_settings
    from app.models.runtime import available_local_sam_models, resolve_sam2_config_path

    settings = get_settings()
    sam31_config_path = settings.models_dir / "sam3.1" / "config.json"
    sam3_config_path = settings.models_dir / "sam3" / "config.json"
    sam2_config_path = resolve_sam2_config_path(settings)
    # Post-bootstrap verification is intentionally stricter than runtime auto-mode:
    # a ready machine must already have at least one complete local SAM asset set.
    sam_local_models = available_local_sam_models(settings)

    sam_checks = [
        {
            "name": "sam3.1 config",
            "path": str(sam31_config_path),
            "exists": sam31_config_path.exists(),
        },
        {
            "name": "sam3.1 checkpoint",
            "path": str(settings.sam_checkpoint_path),
            "exists": settings.sam_checkpoint_path.exists(),
        },
        {
            "name": "sam3 config",
            "path": str(sam3_config_path),
            "exists": sam3_config_path.exists(),
        },
        {
            "name": "sam3 checkpoint",
            "path": str(settings.sam_legacy_checkpoint_path),
            "exists": settings.sam_legacy_checkpoint_path.exists(),
        },
        {
            "name": "sam2.1 checkpoint",
            "path": str(settings.sam2_checkpoint_path),
            "exists": settings.sam2_checkpoint_path.exists(),
        },
    ]

    effecterase_paths = []
    for name, path in settings.effecterase_required_paths().items():
        effecterase_paths.append(
            {
                "name": name,
                "path": str(path),
                "exists": path.exists(),
            }
        )

    return {
        "sam": {
            "ok": len(sam_local_models) > 0,
            "localModels": sam_local_models,
            "checks": sam_checks,
            "sam2Config": {
                "path": str(sam2_config_path or settings.sam2_config_path),
                "exists": sam2_config_path is not None,
                "source": (
                    "configured"
                    if sam2_config_path == settings.sam2_config_path and sam2_config_path is not None
                    else "package"
                    if sam2_config_path is not None
                    else "missing"
                ),
            },
        },
        "effectErase": {
            "ok": all(entry["exists"] for entry in effecterase_paths),
            "requiredPaths": effecterase_paths,
        },
    }


def _aggregate(manager: str, strategy: str, worker_env: str, sam_env: str | None, remove_env: str | None) -> dict:
    if strategy == "shared":
        env_targets = [(worker_env, "shared")]
    else:
        env_targets = [
            (sam_env or worker_env, "sam"),
            (remove_env or "", "remove"),
        ]

    env_reports = []
    for env_name, role in env_targets:
        if not env_name:
            env_reports.append(
                {
                    "envName": env_name,
                    "role": role,
                    "label": PROBE_DEFINITIONS[role]["label"],
                    "imports": {
                        "ok": False,
                        "probe": PROBE_DEFINITIONS[role]["code"],
                        "error": "Environment name is missing.",
                    },
                    "cuda": {
                        "ok": False,
                        "torchImportOk": False,
                        "torchVersion": None,
                        "cudaAvailable": False,
                        "deviceCount": 0,
                        "firstDeviceName": None,
                        "error": "Environment name is missing.",
                    },
                }
            )
            continue
        env_reports.append(_run_probe(manager, worker_env, env_name, role))

    models = _model_report()
    imports_ok = all(report["imports"]["ok"] for report in env_reports)
    cuda_ok = all(report["cuda"]["ok"] for report in env_reports)

    return {
        "ok": imports_ok and cuda_ok and models["sam"]["ok"] and models["effectErase"]["ok"],
        "bootstrap": {
            "envManager": manager,
            "activeStrategy": strategy,
            "workerEnvName": worker_env,
            "samEnvName": sam_env,
            "removeEnvName": remove_env,
            "envNames": [report["envName"] for report in env_reports if report["envName"]],
        },
        "envChecks": env_reports,
        "models": models,
    }


def _print_report(report: dict) -> None:
    print("Bootstrap verification")
    print()
    print("Bootstrap context:")
    print(f"- env manager: {report['bootstrap']['envManager']}")
    print(f"- strategy: {report['bootstrap']['activeStrategy']}")
    print(f"- worker env: {report['bootstrap']['workerEnvName']}")
    print(f"- checked envs: {', '.join(report['bootstrap']['envNames']) or 'none'}")
    print()
    print("CUDA checks:")
    for env_report in report["envChecks"]:
        cuda = env_report["cuda"]
        detail = f"torch {cuda['torchVersion']}" if cuda["torchVersion"] else "torch unavailable"
        if cuda["firstDeviceName"]:
            detail = f"{detail}; {cuda['deviceCount']} device(s); {cuda['firstDeviceName']}"
        else:
            detail = f"{detail}; {cuda['deviceCount']} device(s)"
        if cuda["error"]:
            detail = f"{detail}; {cuda['error']}"
        print(f"- {env_report['envName']} [{env_report['label']}]: {_status(cuda['ok'])} ({detail})")
    print()
    print("Env import checks:")
    for env_report in report["envChecks"]:
        imports = env_report["imports"]
        detail = imports["error"] or "probe passed"
        print(f"- {env_report['envName']} [{env_report['label']}]: {_status(imports['ok'])} ({detail})")
    print()
    print("Model asset checks:")
    sam = report["models"]["sam"]
    local_models = ", ".join(sam["localModels"]) if sam["localModels"] else "none"
    print(f"- SAM runtime set: {_status(sam['ok'])} (local runnable models: {local_models})")
    for check in sam["checks"]:
        print(f"  - {check['name']}: {_status(check['exists'])} ({check['path']})")
    sam2_config = sam["sam2Config"]
    print(
        f"  - sam2.1 config: {_status(sam2_config['exists'])} "
        f"({sam2_config['path']}; source={sam2_config['source']})"
    )
    effecterase = report["models"]["effectErase"]
    print(f"- EffectErase assets: {_status(effecterase['ok'])}")
    for entry in effecterase["requiredPaths"]:
        print(f"  - {entry['name']}: {_status(entry['exists'])} ({entry['path']})")
    print()
    print(f"Final result: {_status(report['ok'])}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify bootstrap readiness for the worker envs.")
    subparsers = parser.add_subparsers(dest="command")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--manager", required=True, choices=("conda", "micromamba"))
    aggregate_parser.add_argument("--strategy", required=True, choices=("shared", "split"))
    aggregate_parser.add_argument("--worker-env", required=True)
    aggregate_parser.add_argument("--sam-env")
    aggregate_parser.add_argument("--remove-env")
    aggregate_parser.add_argument("--json", action="store_true")

    probe_parser = subparsers.add_parser("probe-env")
    probe_parser.add_argument("--role", required=True, choices=tuple(PROBE_DEFINITIONS))
    probe_parser.add_argument("--env-name", required=True)

    args = parser.parse_args(argv)
    if args.command == "probe-env":
        return _probe_env(args.role, args.env_name)

    if args.command != "aggregate":
        parser.print_help()
        return 1

    report = _aggregate(args.manager, args.strategy, args.worker_env, args.sam_env, args.remove_env)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
