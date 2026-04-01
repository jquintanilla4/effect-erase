from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile


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


def _optional_status(ok: bool, required: bool) -> str:
    if ok:
        return "PASS"
    return "FAIL" if required else "WARN"


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


def _checkpoint_check(name: str, path) -> dict:
    entry = {
        "name": name,
        "path": str(path),
        "exists": path.exists(),
        "ok": path.exists(),
        "error": None,
    }
    if not path.exists():
        return entry

    try:
        if not zipfile.is_zipfile(path):
            raise RuntimeError("file is not a zip-based PyTorch checkpoint")
        with zipfile.ZipFile(path) as archive:
            bad_entry = archive.testzip()
            if bad_entry is not None:
                raise RuntimeError(f"archive entry failed CRC validation: {bad_entry}")
    except Exception as error:
        entry["ok"] = False
        entry["error"] = _error_text(error)
    return entry


def _model_report() -> dict:
    from app.core.config import get_settings
    from app.models.runtime import available_local_sam_models, resolve_sam2_config_name, resolve_sam2_config_path

    settings = get_settings()
    sam31_config_path = settings.models_dir / "sam3.1" / "config.json"
    sam3_config_path = settings.models_dir / "sam3" / "config.json"
    sam2_config_path = resolve_sam2_config_path(settings)
    sam2_config_name = resolve_sam2_config_name(settings)
    # The report tracks on-disk asset readiness separately from the final
    # bootstrap verdict so setup can allow staged/manual provisioning when asked.
    sam_local_models = available_local_sam_models(settings)

    sam31_checkpoint = _checkpoint_check("sam3.1 checkpoint", settings.sam_checkpoint_path)
    sam3_checkpoint = _checkpoint_check("sam3 checkpoint", settings.sam_legacy_checkpoint_path)
    sam_local_models = [model for model in available_local_sam_models(settings)]
    if not sam31_checkpoint["ok"] and "sam3.1" in sam_local_models:
        sam_local_models.remove("sam3.1")
    if not sam3_checkpoint["ok"] and "sam3" in sam_local_models:
        sam_local_models.remove("sam3")

    sam_checks = [
        {
            "name": "sam3.1 config",
            "path": str(sam31_config_path),
            "exists": sam31_config_path.exists(),
            "ok": sam31_config_path.exists(),
            "error": None,
        },
        sam31_checkpoint,
        {
            "name": "sam3 config",
            "path": str(sam3_config_path),
            "exists": sam3_config_path.exists(),
            "ok": sam3_config_path.exists(),
            "error": None,
        },
        sam3_checkpoint,
        {
            "name": "sam2.1 checkpoint",
            "path": str(settings.sam2_checkpoint_path),
            "exists": settings.sam2_checkpoint_path.exists(),
            "ok": settings.sam2_checkpoint_path.exists(),
            "error": None,
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
                "configName": sam2_config_name,
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


def _runtime_mode() -> str:
    from app.core.config import get_settings

    settings = get_settings()
    # Mirror the runtime selection logic without importing the heavier runtime
    # module here. Explicit mock mode is the one case where bootstrap should
    # tolerate a host that cannot see CUDA yet.
    if settings.use_mock_runtime:
        return "mock"
    return settings.runtime_mode


def _verification_policy(runtime_mode: str, bootstrap_mode: bool, allow_missing_model_assets: bool) -> dict:
    return {
        "runtimeMode": runtime_mode,
        "bootstrapMode": bootstrap_mode,
        "allowMissingModelAssets": allow_missing_model_assets,
        "cudaRequired": not (bootstrap_mode and runtime_mode == "mock"),
        "modelAssetsRequired": not (bootstrap_mode and allow_missing_model_assets),
    }


def _aggregate(
    manager: str,
    strategy: str,
    worker_env: str,
    sam_env: str | None,
    remove_env: str | None,
    *,
    bootstrap_mode: bool,
    allow_missing_model_assets: bool,
) -> dict:
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

    policy = _verification_policy(_runtime_mode(), bootstrap_mode, allow_missing_model_assets)
    models = _model_report()
    imports_ok = all(report["imports"]["ok"] for report in env_reports)
    cuda_ok = all(report["cuda"]["ok"] for report in env_reports)
    model_assets_ok = models["sam"]["ok"] and models["effectErase"]["ok"]
    real_inference_ready = imports_ok and cuda_ok and model_assets_ok
    bootstrap_compatible = (
        imports_ok
        and (cuda_ok or not policy["cudaRequired"])
        and (model_assets_ok or not policy["modelAssetsRequired"])
    )

    return {
        "ok": bootstrap_compatible,
        "bootstrapCompatible": bootstrap_compatible,
        "realInferenceReady": real_inference_ready,
        "bootstrap": {
            "envManager": manager,
            "activeStrategy": strategy,
            "workerEnvName": worker_env,
            "samEnvName": sam_env,
            "removeEnvName": remove_env,
            "envNames": [report["envName"] for report in env_reports if report["envName"]],
        },
        "policy": policy,
        "checks": {
            "importsOk": imports_ok,
            "cudaOk": cuda_ok,
            "modelAssetsOk": model_assets_ok,
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
    print(f"- runtime mode: {report['policy']['runtimeMode']}")
    print(f"- bootstrap mode: {'yes' if report['policy']['bootstrapMode'] else 'no'}")
    print(f"- cuda required: {'yes' if report['policy']['cudaRequired'] else 'no'}")
    print(f"- model assets required: {'yes' if report['policy']['modelAssetsRequired'] else 'no'}")
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
        print(
            f"- {env_report['envName']} [{env_report['label']}]: "
            f"{_optional_status(cuda['ok'], report['policy']['cudaRequired'])} ({detail})"
        )
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
    print(
        f"- SAM runtime set: "
        f"{_optional_status(sam['ok'], report['policy']['modelAssetsRequired'])} "
        f"(local runnable models: {local_models})"
    )
    for check in sam["checks"]:
        detail = check["path"]
        if check["error"]:
            detail = f"{detail}; {check['error']}"
        print(f"  - {check['name']}: {_status(check['ok'])} ({detail})")
    sam2_config = sam["sam2Config"]
    print(
        f"  - sam2.1 config: {_status(sam2_config['exists'])} "
        f"({sam2_config['path']}; source={sam2_config['source']}; config_name={sam2_config['configName'] or 'missing'})"
    )
    effecterase = report["models"]["effectErase"]
    print(
        f"- EffectErase assets: "
        f"{_optional_status(effecterase['ok'], report['policy']['modelAssetsRequired'])}"
    )
    for entry in effecterase["requiredPaths"]:
        print(f"  - {entry['name']}: {_status(entry['exists'])} ({entry['path']})")
    print()
    print(f"Bootstrap-compatible: {_status(report['bootstrapCompatible'])}")
    print(f"Real inference ready: {_status(report['realInferenceReady'])}")
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
    aggregate_parser.add_argument("--bootstrap-mode", action="store_true")
    aggregate_parser.add_argument("--allow-missing-model-assets", action="store_true")
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

    report = _aggregate(
        args.manager,
        args.strategy,
        args.worker_env,
        args.sam_env,
        args.remove_env,
        bootstrap_mode=args.bootstrap_mode,
        allow_missing_model_assets=args.allow_missing_model_assets,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
