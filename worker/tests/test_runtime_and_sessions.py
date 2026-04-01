import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import unittest
from contextlib import contextmanager
from unittest.mock import patch

from fastapi import HTTPException
import numpy as np

# Repo-root discovery does not put worker/ on sys.path, so add it here before
# importing the worker package under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings
import app.models.runtime as runtime_module
from app.models.runtime import RealSamRuntime, SessionRuntimeState
from app.models.video import VideoMetadata
from app.schemas.api import StartSessionRequest
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
                SimpleNamespace(sessionId="session-1", frameIndex=0, points=[SimpleNamespace(x=0.5, y=0.5, label="positive")])
            )

        self.assertEqual(context.exception.status_code, 503)
        self.assertEqual(context.exception.detail, "sam3.1 prompt failed")

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
            service.propagate(SimpleNamespace(sessionId="session-1"))

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.detail, "No prompt exists for propagation.")


if __name__ == "__main__":
    unittest.main()
