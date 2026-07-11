from __future__ import annotations

import base64
from pathlib import Path

import pytest


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
GIF_BYTES = b"GIF89a" + b"\x00" * 16
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 16


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _patch_paths(monkeypatch, app_settings, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    monkeypatch.setattr(app_settings, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(app_settings, "APP_SETTINGS_PATH", project_root / "data" / "app_settings.json")
    monkeypatch.setattr(app_settings, "SIDEBAR_IMAGE_PATH", project_root / "data" / "ui" / "sidebar-image.bin")
    monkeypatch.setattr(app_settings, "SIDEBAR_IMAGE_META_PATH", project_root / "data" / "ui" / "sidebar-image.json")


def test_app_settings_default_sidebar_image_url_is_blank():
    from services.rag_api.app_settings import AppSettings

    assert AppSettings().sidebar_image_url == ""


def test_frontend_uses_crab_as_default_and_persisted_image_as_override():
    source = Path("apps/web/src/pages/ChatPage.tsx").read_text(encoding="utf-8")
    bundle = next(Path("apps/web/dist/assets").glob("index-*.js")).read_text(encoding="utf-8")

    assert "function sidebarDefaultImage(){return`/picture/crab.png`}" in source
    assert "/picture/crab.png" in bundle
    assert "1-clean.gif" not in bundle
    assert "2-clean.gif" not in bundle
    assert "sidebar_image_url" in source
    assert "?v=${Date.now()}" in source
    assert Path("apps/web/dist/picture/crab.png").is_file()


def test_save_sidebar_image_writes_binary_metadata_and_updates_settings(tmp_path: Path, monkeypatch):
    from services.rag_api import app_settings

    _patch_paths(monkeypatch, app_settings, tmp_path)

    saved = app_settings.save_sidebar_image(
        app_settings.SidebarImageUpload(
            filename="cover.png",
            content_type="image/png",
            data_base64=_b64(PNG_BYTES),
        )
    )

    assert saved.sidebar_image_url == "/api/app-assets/sidebar-image"
    assert app_settings.SIDEBAR_IMAGE_PATH.read_bytes() == PNG_BYTES
    assert "image/png" in app_settings.SIDEBAR_IMAGE_META_PATH.read_text(encoding="utf-8")


def test_save_sidebar_image_accepts_jprg_when_file_header_is_jpeg(tmp_path: Path, monkeypatch):
    from services.rag_api import app_settings

    _patch_paths(monkeypatch, app_settings, tmp_path)

    saved = app_settings.save_sidebar_image(
        app_settings.SidebarImageUpload(
            filename="cover.jprg",
            content_type="image/jpeg",
            data_base64=_b64(JPEG_BYTES),
        )
    )

    assert saved.sidebar_image_url == "/api/app-assets/sidebar-image"
    assert app_settings.read_sidebar_image_asset()[1] == "image/jpeg"


def test_save_sidebar_image_rejects_disguised_image(tmp_path: Path, monkeypatch):
    from services.rag_api import app_settings

    _patch_paths(monkeypatch, app_settings, tmp_path)

    with pytest.raises(ValueError, match="图片格式"):
        app_settings.save_sidebar_image(
            app_settings.SidebarImageUpload(
                filename="bad.png",
                content_type="image/png",
                data_base64=_b64(b"not an image"),
            )
        )


def test_save_sidebar_image_rejects_too_large_file(tmp_path: Path, monkeypatch):
    from services.rag_api import app_settings

    _patch_paths(monkeypatch, app_settings, tmp_path)

    with pytest.raises(ValueError, match="10 MB"):
        app_settings.save_sidebar_image(
            app_settings.SidebarImageUpload(
                filename="large.gif",
                content_type="image/gif",
                data_base64=_b64(GIF_BYTES + b"x" * (10 * 1024 * 1024)),
            )
        )


def test_read_sidebar_image_asset_returns_content_type_or_raises_404(tmp_path: Path, monkeypatch):
    from fastapi import HTTPException
    from services.rag_api import app_settings

    _patch_paths(monkeypatch, app_settings, tmp_path)

    with pytest.raises(HTTPException) as missing:
        app_settings.read_sidebar_image_asset()
    assert missing.value.status_code == 404

    app_settings.save_sidebar_image(
        app_settings.SidebarImageUpload(
            filename="cover.gif",
            content_type="image/gif",
            data_base64=_b64(GIF_BYTES),
        )
    )

    data, content_type = app_settings.read_sidebar_image_asset()

    assert data == GIF_BYTES
    assert content_type == "image/gif"
