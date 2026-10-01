"""Acquisition preserves pinned sources and can resume partial downloads."""

from __future__ import annotations

import hashlib
import pathlib

import httpx
import pytest

from ec_analysis.data.acquisition import SourceDownloader, SourceManifest
from ec_analysis.data.sources import SourceAsset


def make_source(filename: str, payload: bytes) -> SourceAsset:
    return SourceAsset(
        filename=filename,
        source_url=f"https://source.example/{filename}",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_verified_sources_are_reused_without_network(tmp_path: pathlib.Path) -> None:
    payload = b"official fixture bytes"
    source_asset = make_source("sample.zip", payload)
    requests: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        downloader = SourceDownloader(http_client=http_client)
        first_manifest = downloader.download_assets((source_asset,), tmp_path)
        second_manifest = downloader.download_assets((source_asset,), tmp_path)

    assert len(requests) == 1
    assert (tmp_path / "sample.zip").read_bytes() == payload
    assert first_manifest.assets == second_manifest.assets
    assert second_manifest.source_kind == "format_sample"


@pytest.mark.parametrize("maximum_bytes, expected_error", [(3, "download limit"), (100, "checksum mismatch")])
def test_invalid_download_is_not_published(tmp_path: pathlib.Path, maximum_bytes: int, expected_error: str) -> None:
    source_asset = make_source("sample.zip", b"expected data")
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"wrong data"))
    ) as client:
        with pytest.raises(ValueError, match=expected_error):
            SourceDownloader(http_client=client, maximum_bytes=maximum_bytes).download_assets((source_asset,), tmp_path)

    assert not (tmp_path / "sample.zip").exists()
    assert not list(tmp_path.glob("*.part"))


def test_partial_acquisition_retains_provenance(tmp_path: pathlib.Path) -> None:
    first_asset = make_source("first.zip", b"first")
    second_asset = make_source("second.pdf", b"second")

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/first.zip":
            return httpx.Response(200, content=b"first")
        return httpx.Response(503)

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        with pytest.raises(httpx.HTTPStatusError):
            SourceDownloader(http_client=http_client).download_assets((first_asset, second_asset), tmp_path)

    saved_manifest = SourceManifest.model_validate_json((tmp_path / "manifest.json").read_text())
    assert [asset.filename for asset in saved_manifest.assets] == ["first.zip"]
    assert (tmp_path / "first.zip").read_bytes() == b"first"


def test_existing_modified_source_is_preserved(tmp_path: pathlib.Path) -> None:
    source_asset = make_source("sample.zip", b"expected data")
    (tmp_path / "sample.zip").write_bytes(b"local changed data")
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(500))) as http_client:
        with pytest.raises(ValueError, match="Existing source checksum mismatch"):
            SourceDownloader(http_client=http_client).download_assets((source_asset,), tmp_path)

    assert (tmp_path / "sample.zip").read_bytes() == b"local changed data"
