import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import tempfile
import unittest
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
import numpy as np

# Repo-root discovery does not put worker/ on sys.path, so add it here before
# importing the worker package under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings
import app.models.runtime as runtime_module
from app.models.runtime import RealEffectEraseRuntime, RealSamRuntime, SessionRuntimeState
from app.models.video import VideoMetadata
from app.schemas.api import StartSessionRequest
from app.services.jobs import JobService
from app.services.sessions import SessionService


class SamSettingsTests(unittest.TestCase):
    def test_default_sam31_multiplex_count_matches_checkpoint_shape(self):
        settings = Settings()
        self.assertEqual(settings.sam_multiplex_count, 16)

    def test_sam31_multiplex_count_can_be_overridden_from_env(self):
        with patch.dict(os.environ, {"WORKER_SAM_MULTIPLEX_COUNT": "8"}):
            settings = Settings()

        self.assertEqual(settings.sam_multiplex_count, 8)


class RealSamRuntimeFallbackTests(unittest.TestCase):
    def test_patch_sam31_partial_propagation_output_keeps_refined_masks_without_cache(self):
        class FakeTracking:
            _effecterase_partial_output_patch_applied = False

            def _build_sam2_output(self, inference_state, frame_idx, refined_obj_id_to_mask=None):
                return {"should_not": "survive"}

        fake_module = ModuleType("sam3.model.sam3_multiplex_tracking")
        fake_module.Sam3MultiplexTrackingWithInteractivity = FakeTracking

        with patch.dict(sys.modules, {"sam3.model.sam3_multiplex_tracking": fake_module}):
            runtime_module._patch_sam31_partial_propagation_output()

        tracker = FakeTracking()
        out_without_cache = tracker._build_sam2_output(
            {"cached_frame_outputs": {}},
            3,
            refined_obj_id_to_mask={1: "mask-1"},
        )
        out_with_cache = tracker._build_sam2_output(
            {"cached_frame_outputs": {3: {2: "mask-2"}}},
            3,
            refined_obj_id_to_mask={1: "mask-1"},
        )

        self.assertEqual(out_without_cache, {1: "mask-1"})
        self.assertEqual(out_with_cache, {2: "mask-2", 1: "mask-1"})
        self.assertTrue(FakeTracking._effecterase_partial_output_patch_applied)

    def test_build_sam2_predictor_uses_package_relative_config_name(self):
        checkpoint_path = Path(__file__).resolve()
        settings = SimpleNamespace(
            sam2_checkpoint_path=checkpoint_path,
            sam2_allow_hf_download=False,
            sam2_hf_model_id="facebook/sam2.1-hiera-base-plus",
        )
        runtime = RealSamRuntime(settings)

        with (
            patch.object(runtime, "_sam2_config_path", return_value=Path("/tmp/configs/sam2.1_hiera_b+.yaml")),
            patch.object(runtime, "_sam2_config_name", return_value="configs/sam2.1/sam2.1_hiera_b+.yaml"),
            patch("sam2.build_sam.build_sam2_video_predictor", return_value="predictor") as build_predictor,
        ):
            predictor = runtime._build_sam2_predictor()

        self.assertEqual(predictor, "predictor")
        build_predictor.assert_called_once_with(
            "configs/sam2.1/sam2.1_hiera_b+.yaml",
            checkpoint_path.as_posix(),
        )

    def test_predictor_uses_updated_sam31_multiplex_default(self):
        settings = SimpleNamespace(
            sam_compile=False,
            sam_async_loading_frames=False,
            sam_max_num_objects=1,
            sam_multiplex_count=16,
        )
        runtime = RealSamRuntime(settings)

        with (
            patch.object(runtime, "_checkpoint_path", return_value="/tmp/sam3.1_multiplex.pt"),
            patch.object(runtime, "_sam3_use_fa3", return_value=(False, "FA3 disabled for test")),
            patch("sam3.model_builder.build_sam3_predictor", return_value="predictor") as build_predictor,
        ):
            predictor = runtime._predictor("sam3.1")

        self.assertEqual(predictor, "predictor")
        build_predictor.assert_called_once_with(
            checkpoint_path="/tmp/sam3.1_multiplex.pt",
            version="sam3.1",
            compile=False,
            async_loading_frames=False,
            max_num_objects=1,
            multiplex_count=16,
            use_fa3=False,
            use_rope_real=False,
        )

    def test_predictor_enables_fa3_on_supported_gpu(self):
        settings = SimpleNamespace(
            sam_compile=False,
            sam_async_loading_frames=False,
            sam_max_num_objects=1,
            sam_multiplex_count=16,
        )
        runtime = RealSamRuntime(settings)

        with (
            patch.object(runtime, "_checkpoint_path", return_value="/tmp/sam3.1_multiplex.pt"),
            patch.object(runtime, "_sam3_use_fa3", return_value=(True, "FA3 enabled for test")),
            patch("sam3.model_builder.build_sam3_predictor", return_value="predictor") as build_predictor,
        ):
            runtime._predictor("sam3.1")

        self.assertEqual(build_predictor.call_args.kwargs["use_fa3"], True)
        self.assertEqual(build_predictor.call_args.kwargs["use_rope_real"], False)

    def test_predictor_uses_real_rope_when_compile_is_enabled(self):
        settings = SimpleNamespace(
            sam_compile=True,
            sam_async_loading_frames=False,
            sam_max_num_objects=1,
            sam_multiplex_count=16,
        )
        runtime = RealSamRuntime(settings)

        with (
            patch.object(runtime, "_checkpoint_path", return_value="/tmp/sam3.1_multiplex.pt"),
            patch.object(runtime, "_sam3_use_fa3", return_value=(False, "FA3 disabled for test")),
            patch("sam3.model_builder.build_sam3_predictor", return_value="predictor") as build_predictor,
        ):
            runtime._predictor("sam3.1")

        self.assertEqual(build_predictor.call_args.kwargs["use_rope_real"], True)

    def test_start_raises_clear_error_when_requested_predictor_fails(self):
        runtime = RealSamRuntime(
            SimpleNamespace(
                sam2_checkpoint_path=Path("/tmp/sam2.1_hiera_base_plus.pt"),
                sam_checkpoint_for_model=lambda _model: Path("/tmp/sam3.1_multiplex.pt"),
            )
        )
        video_path = Path("/tmp/source.mp4")

        with (
            patch("app.models.runtime.load_video_metadata", return_value=VideoMetadata(video_path, 832, 480, 24.0, 12)),
            patch("app.models.runtime.available_sam_models", return_value=["sam3.1", "sam2.1"]),
            patch.object(runtime, "_resolved_model_name", return_value="sam3.1"),
            patch.object(runtime, "_predictor", side_effect=RuntimeError("corrupt checkpoint")),
        ):
            with self.assertRaises(RuntimeError) as context:
                runtime.start("project-1", video_path, "sam3.1")

        self.assertIn("Failed to initialize sam3.1", str(context.exception))
        self.assertIn("corrupt checkpoint", str(context.exception))

    def test_start_raises_when_requested_model_is_unavailable(self):
        runtime = RealSamRuntime(SimpleNamespace())

        with patch("app.models.runtime.available_sam_models", return_value=["sam2.1"]):
            with self.assertRaises(RuntimeError) as context:
                runtime._resolved_model_name("sam3.1")

        self.assertIn("Requested SAM model 'sam3.1' is not available", str(context.exception))

    def test_sam31_add_prompt_reenters_request_autocast_context(self):
        settings = SimpleNamespace()
        runtime = RealSamRuntime(settings)
        state = SessionRuntimeState(
            project_id="project-1",
            source_video_path=Path("/tmp/project-1.mp4"),
            frame_count=12,
            width=832,
            height=480,
            fps=24.0,
            model_name="sam3.1",
            backend_state="session-1",
        )
        predictor = SimpleNamespace(handle_request=lambda _request: {"outputs": None})
        context_events = []

        @contextmanager
        def fake_request_context(model_name):
            context_events.append(("enter", model_name))
            try:
                yield
            finally:
                context_events.append(("exit", model_name))

        with (
            patch.object(runtime, "_predictor", return_value=predictor),
            patch.object(runtime, "_sam3_request_context", side_effect=fake_request_context),
            patch("app.models.runtime.read_frame", return_value=np.zeros((480, 832, 3), dtype=np.uint8)),
            patch("app.models.runtime._save_preview_assets"),
        ):
            runtime.add_prompt(
                state=state,
                frame_index=0,
                points=[],
                output_mask_path=Path("/tmp/mask.png"),
                output_frame_path=Path("/tmp/frame.png"),
            )

        self.assertEqual(
            context_events,
            [("enter", "sam3.1"), ("exit", "sam3.1")],
        )

    def test_sam31_start_session_reenters_request_autocast_context(self):
        runtime = RealSamRuntime(SimpleNamespace())
        predictor = SimpleNamespace(
            handle_request=lambda _request: {"session_id": "session-1"},
        )
        context_events = []

        @contextmanager
        def fake_request_context(model_name):
            context_events.append(("enter", model_name))
            try:
                yield
            finally:
                context_events.append(("exit", model_name))

        with patch.object(runtime, "_sam3_request_context", side_effect=fake_request_context):
            session_id = runtime._start_backend_state(predictor, "sam3.1", Path("/tmp/source.mp4"))

        self.assertEqual(session_id, "session-1")
        self.assertEqual(
            context_events,
            [("enter", "sam3.1"), ("exit", "sam3.1")],
        )

    def test_release_resources_closes_session_and_unloads_predictors(self):
        runtime = RealSamRuntime(SimpleNamespace())
        state = SessionRuntimeState(
            project_id="project-1",
            source_video_path=Path("/tmp/project-1.mp4"),
            frame_count=12,
            width=832,
            height=480,
            fps=24.0,
            model_name="sam3.1",
            backend_state="session-1",
        )
        predictor = MagicMock()
        runtime.predictors["sam3.1"] = predictor
        context_events = []

        @contextmanager
        def fake_request_context(model_name):
            context_events.append(("enter", model_name))
            try:
                yield
            finally:
                context_events.append(("exit", model_name))

        with (
            patch.object(runtime, "_sam3_request_context", side_effect=fake_request_context),
            patch("app.models.runtime.gc.collect") as gc_collect,
            patch("app.models.runtime._clear_cuda_runtime_memory") as clear_cuda_runtime_memory,
        ):
            runtime.release_resources(state)

        predictor.handle_request.assert_called_once_with({"type": "close_session", "session_id": "session-1"})
        predictor.shutdown.assert_called_once_with()
        self.assertEqual(runtime.predictors, {})
        self.assertIsNone(state.backend_state)
        self.assertEqual(context_events, [("enter", "sam3.1"), ("exit", "sam3.1")])
        gc_collect.assert_called_once_with()
        clear_cuda_runtime_memory.assert_called_once_with()


class RealEffectEraseRuntimeTests(unittest.TestCase):
    def test_remove_uses_source_clip_length_for_num_frames(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            required = {}
            for name in ("lora", "text_encoder", "vae", "dit", "image_encoder"):
                asset_path = temp_path / f"{name}.bin"
                asset_path.touch()
                required[name] = asset_path

            output_path = temp_path / "removed_output.mp4"
            output_path.touch()
            settings = SimpleNamespace(
                bootstrap_state_path=temp_path / "bootstrap-status.json",
                effecterase_num_frames=81,
                effecterase_frame_interval=1,
                default_height=480,
                default_width=832,
                effecterase_seed=2025,
                effecterase_cfg=1.0,
                effecterase_lora_alpha=1.0,
                effecterase_num_inference_steps=50,
                effecterase_tiled=False,
                effecterase_use_teacache=False,
                root_dir=temp_path,
                effecterase_required_paths=lambda: required,
            )
            bootstrap_status = SimpleNamespace(
                activeStrategy="split",
                removeEnvName="effecterase-remove-clean",
                workerEnvName="effecterase-sam-clean",
                envManager="conda",
            )

            with patch("app.models.runtime.load_bootstrap_status", return_value=bootstrap_status):
                runtime = RealEffectEraseRuntime(settings)

            source_path = temp_path / "source.mp4"
            mask_path = temp_path / "mask_sequence.mp4"
            source_path.touch()
            mask_path.touch()

            metadata_by_path = {
                source_path: VideoMetadata(source_path, 832, 480, 24.0, 56),
                mask_path: VideoMetadata(mask_path, 832, 480, 24.0, 56),
                output_path: VideoMetadata(output_path, 832, 480, 24.0, 56),
            }

            with (
                patch("app.models.runtime.load_video_metadata", side_effect=lambda path: metadata_by_path[path]),
                patch("app.models.runtime.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr="")) as run_mock,
            ):
                runtime.remove(
                    source_video_path=source_path,
                    mask_video_path=mask_path,
                    output_video_path=output_path,
                    progress_callback=lambda _value: None,
                )

        command = run_mock.call_args.args[0]
        num_frames_index = command.index("--num_frames") + 1
        self.assertEqual(command[num_frames_index], "56")

    def test_remove_rejects_mask_videos_with_different_frame_count(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            settings = SimpleNamespace(
                bootstrap_state_path=temp_path / "bootstrap-status.json",
                effecterase_num_frames=81,
            )
            bootstrap_status = SimpleNamespace(
                activeStrategy="split",
                removeEnvName="effecterase-remove-clean",
                workerEnvName="effecterase-sam-clean",
                envManager="conda",
            )

            with patch("app.models.runtime.load_bootstrap_status", return_value=bootstrap_status):
                runtime = RealEffectEraseRuntime(settings)

            source_path = temp_path / "source.mp4"
            mask_path = temp_path / "mask_sequence.mp4"
            source_path.touch()
            mask_path.touch()

            metadata_by_path = {
                source_path: VideoMetadata(source_path, 832, 480, 24.0, 56),
                mask_path: VideoMetadata(mask_path, 832, 480, 24.0, 48),
            }

            with patch("app.models.runtime.load_video_metadata", side_effect=lambda path: metadata_by_path[path]):
                with self.assertRaises(RuntimeError) as context:
                    runtime.remove(
                        source_video_path=source_path,
                        mask_video_path=mask_path,
                        output_video_path=temp_path / "removed_output.mp4",
                        progress_callback=lambda _value: None,
                    )

        self.assertIn("Source frames=56, mask frames=48", str(context.exception))


class JobServiceTests(unittest.TestCase):
    def test_create_removal_job_releases_sam_resources_before_spawning_remove_thread(self):
        settings = SimpleNamespace(public_base_url="http://localhost:8000")
        project_service = SimpleNamespace(
            require_source_video=lambda project_id: Path(f"/tmp/{project_id}.mp4"),
            storage=SimpleNamespace(
                project_dir=lambda project_id: Path(f"/tmp/{project_id}"),
                artifact_url=lambda base_url, path: f"{base_url}/artifacts/{Path(path).name}",
            ),
        )
        runtime_state = SessionRuntimeState(
            project_id="project-1",
            source_video_path=Path("/tmp/project-1.mp4"),
            frame_count=12,
            width=832,
            height=480,
            fps=24.0,
            model_name="sam3.1",
            backend_state="session-1",
        )
        session_service = SimpleNamespace(
            require_mask_video=lambda session_id: (runtime_state, Path("/tmp/project-1/mask_sequence.mp4")),
            release_runtime_resources=MagicMock(),
        )
        fake_thread = MagicMock()

        with (
            patch("app.services.jobs.build_remove_runtime", return_value=SimpleNamespace()),
            patch("app.services.jobs.threading.Thread", return_value=fake_thread),
        ):
            service = JobService(settings, project_service, session_service)
            response = service.create_removal_job(
                SimpleNamespace(projectId="project-1", sessionId="session-1"),
                None,
                "http://gpu-box.tailnet-name.ts.net:8000",
            )

        session_service.release_runtime_resources.assert_called_once_with("session-1")
        fake_thread.start.assert_called_once_with()
        self.assertEqual(response.projectId, "project-1")
        self.assertEqual(response.status, "queued")
        self.assertIsNone(response.resultUrl)

    def test_get_job_builds_result_url_from_polling_origin(self):
        settings = SimpleNamespace(public_base_url="http://localhost:8000")
        project_service = SimpleNamespace(
            require_source_video=lambda project_id: Path(f"/tmp/{project_id}.mp4"),
            storage=SimpleNamespace(
                project_dir=lambda project_id: Path(f"/tmp/{project_id}"),
                artifact_url=lambda base_url, path: f"{base_url}/artifacts/{Path(path).name}",
            ),
        )
        session_service = SimpleNamespace(require_mask_video=lambda session_id: None, release_runtime_resources=MagicMock())

        with patch("app.services.jobs.build_remove_runtime", return_value=SimpleNamespace()):
            service = JobService(settings, project_service, session_service)

        service.jobs["job-1"] = SimpleNamespace(
            job_id="job-1",
            project_id="project-1",
            status="completed",
            progress=1.0,
            result_path=Path("/tmp/project-1/removed_output.mp4"),
            error=None,
        )

        response = service.get_job("job-1", "https://abc123xyz-8000.proxy.runpod.net")

        self.assertEqual(
            response.resultUrl,
            "https://abc123xyz-8000.proxy.runpod.net/artifacts/removed_output.mp4",
        )


class SessionServiceStartTests(unittest.TestCase):
    def _service(self):
        settings = SimpleNamespace(public_base_url="http://localhost:8000")
        project_service = SimpleNamespace(
            require_source_video=lambda project_id: Path(f"/tmp/{project_id}.mp4"),
            storage=SimpleNamespace(
                project_dir=lambda project_id: Path(f"/tmp/{project_id}"),
                artifact_url=lambda base_url, path: f"{base_url}/artifacts/{Path(path).name}",
            ),
        )
        with patch("app.services.sessions.build_sam_runtime", return_value=SimpleNamespace()):
            return SessionService(settings, project_service)

    def test_start_session_maps_runtime_error_to_http_503(self):
        service = self._service()
        service.runtime = SimpleNamespace(start=lambda *_args: (_ for _ in ()).throw(RuntimeError("checkpoint corrupt")))

        with self.assertRaises(HTTPException) as context:
            service.start_session(StartSessionRequest(projectId="project-1", model="sam3.1"))

        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.detail, "checkpoint corrupt")

    def test_start_session_returns_runtime_model_name(self):
        service = self._service()
        service.runtime = SimpleNamespace(
            start=lambda *_args: SessionRuntimeState(
                project_id="project-1",
                source_video_path=Path("/tmp/project-1.mp4"),
                frame_count=12,
                width=832,
                height=480,
                fps=24.0,
                model_name="sam2.1",
                backend_state="session-1",
            )
        )

        response = service.start_session(StartSessionRequest(projectId="project-1", model="sam3.1"))

        self.assertEqual(response.projectId, "project-1")
        self.assertEqual(response.model, "sam2.1")
        self.assertEqual(response.frameCount, 12)

    def test_add_prompt_maps_runtime_error_to_http_503(self):
        service = self._service()
        runtime_state = SessionRuntimeState(
            project_id="project-1",
            source_video_path=Path("/tmp/project-1.mp4"),
            frame_count=12,
            width=832,
            height=480,
            fps=24.0,
            model_name="sam3.1",
            backend_state="session-1",
        )
        service.sessions["session-1"] = runtime_state
        service.runtime = SimpleNamespace(
            add_prompt=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("sam3.1 prompt failed"))
        )

        with self.assertRaises(HTTPException) as context:
            service.add_prompt(
                SimpleNamespace(sessionId="session-1", frameIndex=0, points=[SimpleNamespace(x=0.5, y=0.5, label="positive")]),
                "http://gpu-box.tailnet-name.ts.net:8000",
            )

        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.detail, "sam3.1 prompt failed")

    def test_add_prompt_uses_request_origin_for_preview_urls(self):
        service = self._service()
        runtime_state = SessionRuntimeState(
            project_id="project-1",
            source_video_path=Path("/tmp/project-1.mp4"),
            frame_count=12,
            width=832,
            height=480,
            fps=24.0,
            model_name="sam3.1",
            backend_state="session-1",
        )
        service.sessions["session-1"] = runtime_state
        service.runtime = SimpleNamespace(add_prompt=lambda **_kwargs: None)

        response = service.add_prompt(
            SimpleNamespace(sessionId="session-1", frameIndex=3, points=[SimpleNamespace(x=0.5, y=0.5, label="positive")]),
            "http://gpu-box.tailnet-name.ts.net:8000",
        )

        self.assertEqual(response.frameUrl, "http://gpu-box.tailnet-name.ts.net:8000/artifacts/frame_3.png")
        self.assertEqual(response.maskUrl, "http://gpu-box.tailnet-name.ts.net:8000/artifacts/mask_preview_3.png")

    def test_propagate_maps_validation_error_to_http_400(self):
        service = self._service()
        runtime_state = SessionRuntimeState(
            project_id="project-1",
            source_video_path=Path("/tmp/project-1.mp4"),
            frame_count=12,
            width=832,
            height=480,
            fps=24.0,
            model_name="sam3.1",
            backend_state="session-1",
        )
        service.sessions["session-1"] = runtime_state
        service.runtime = SimpleNamespace(propagate=lambda *_args: (_ for _ in ()).throw(ValueError("No prompt exists for propagation.")))

        with self.assertRaises(HTTPException) as context:
            service.propagate(SimpleNamespace(sessionId="session-1"), "http://gpu-box.tailnet-name.ts.net:8000")

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.detail, "No prompt exists for propagation.")

    def test_propagate_uses_request_origin_for_mask_video_url(self):
        service = self._service()
        runtime_state = SessionRuntimeState(
            project_id="project-1",
            source_video_path=Path("/tmp/project-1.mp4"),
            frame_count=12,
            width=832,
            height=480,
            fps=24.0,
            model_name="sam3.1",
            backend_state="session-1",
        )
        service.sessions["session-1"] = runtime_state
        service.runtime = SimpleNamespace(propagate=lambda *_args: SimpleNamespace(frame_count=12))

        response = service.propagate(SimpleNamespace(sessionId="session-1"), "https://abc123xyz-8000.proxy.runpod.net")

        self.assertEqual(
            response.maskVideoUrl,
            "https://abc123xyz-8000.proxy.runpod.net/artifacts/mask_sequence.mp4",
        )


if __name__ == "__main__":
    unittest.main()
