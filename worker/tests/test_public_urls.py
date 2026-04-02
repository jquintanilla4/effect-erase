from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

# Repo-root discovery does not put worker/ on sys.path, so add it here before
# importing the worker package under test.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.public_urls import public_base_url
from app.services.projects import ProjectService


class PublicUrlTests(unittest.TestCase):
    def test_public_base_url_prefers_forwarded_headers(self):
        request = SimpleNamespace(
            headers={
                "host": "127.0.0.1:8000",
                "x-forwarded-proto": "https",
                "x-forwarded-host": "abc123xyz-8000.proxy.runpod.net",
            },
            url=SimpleNamespace(scheme="http", netloc="127.0.0.1:8000"),
        )

        self.assertEqual(public_base_url(request), "https://abc123xyz-8000.proxy.runpod.net")

    def test_public_base_url_falls_back_to_request_host(self):
        request = SimpleNamespace(
            headers={"host": "100.117.141.74:8000"},
            url=SimpleNamespace(scheme="http", netloc="100.117.141.74:8000"),
        )

        self.assertEqual(public_base_url(request), "http://100.117.141.74:8000")


class ProjectServiceUrlTests(unittest.IsolatedAsyncioTestCase):
    async def test_project_urls_use_calling_origin(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = SimpleNamespace(
                bootstrap_state_path=Path(temp_dir) / "bootstrap-status.json",
                projects_dir=Path(temp_dir) / "projects",
            )
            settings.bootstrap_state_path.write_text(
                '{"status":"ready","envManager":"conda","envNames":["effecterase-sam"],'
                '"activeStrategy":"split","workerEnvName":"effecterase-sam"}'
            )
            service = ProjectService(settings)

            created = service.create_project(
                SimpleNamespace(profileId="tailscale", label="ball-cut01.mp4"),
                "http://gpu-box.tailnet-name.ts.net:8000",
            )

            with patch("app.services.projects.load_video_metadata", return_value=SimpleNamespace(width=832, height=480, fps=23.98, frame_count=56)):
                uploaded = await service.save_upload(
                    created.projectId,
                    SimpleNamespace(read=AsyncMock(return_value=b"video-bytes")),
                    "https://abc123xyz-8000.proxy.runpod.net",
                )

        self.assertTrue(created.projectUrl.startswith("http://gpu-box.tailnet-name.ts.net:8000/artifacts/"))
        self.assertTrue(uploaded.sourceUrl.startswith("https://abc123xyz-8000.proxy.runpod.net/artifacts/"))


if __name__ == "__main__":
    unittest.main()
