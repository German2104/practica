"""Download immutable source assets with bounded transfers and provenance."""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import pathlib
import tempfile
import typing

import httpx
from pydantic import BaseModel

from ec_analysis.data.sources import SourceAsset


class AcquiredAsset(BaseModel):
    filename: str
    source_url: str
    sha256: str
    size_bytes: int
    retrieved_at: str | None


class SourceManifest(BaseModel):
    source_id: str = "freddie-mac-release47-public-format-sample"
    source_kind: str = "format_sample"
    verified_at: str
    assets: list[AcquiredAsset]


def file_checksum(input_path: pathlib.Path) -> str:
    with input_path.open("rb") as input_stream:
        return hashlib.file_digest(input_stream, "sha256").hexdigest()


@typing.final
@dataclasses.dataclass(kw_only=True, frozen=True, slots=True)
class SourceDownloader:
    http_client: httpx.Client
    maximum_bytes: int = 20_000_000
    source_id: str = "freddie-mac-release47-public-format-sample"
    source_kind: str = "format_sample"

    def download_assets(self, assets: tuple[SourceAsset, ...], output_directory: pathlib.Path) -> SourceManifest:
        output_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = output_directory / "manifest.json"
        previous_assets = self._previous_assets(manifest_path)
        manifest = SourceManifest(
            source_id=self.source_id,
            source_kind=self.source_kind,
            verified_at=datetime.datetime.now(datetime.UTC).isoformat(),
            assets=[],
        )
        for source_asset in assets:
            manifest.assets.append(self._acquire_asset(source_asset, output_directory, previous_assets))
            manifest_path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return manifest

    def _previous_assets(self, manifest_path: pathlib.Path) -> dict[str, AcquiredAsset]:
        if not manifest_path.exists():
            return {}
        previous_manifest = SourceManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        return {asset.filename: asset for asset in previous_manifest.assets}

    def _acquire_asset(
        self,
        source_asset: SourceAsset,
        output_directory: pathlib.Path,
        previous_assets: dict[str, AcquiredAsset],
    ) -> AcquiredAsset:
        if pathlib.Path(source_asset.filename).name != source_asset.filename:
            raise ValueError("Asset filename must not contain directories")
        destination = output_directory / source_asset.filename
        if destination.exists():
            if file_checksum(destination) != source_asset.expected_sha256:
                raise ValueError(f"Existing source checksum mismatch: {source_asset.filename}; file preserved")
            previous_asset = previous_assets.get(source_asset.filename)
            if previous_asset is not None and previous_asset.sha256 == source_asset.expected_sha256:
                return previous_asset
            return AcquiredAsset(
                filename=source_asset.filename,
                source_url=source_asset.source_url,
                sha256=source_asset.expected_sha256,
                size_bytes=destination.stat().st_size,
                retrieved_at=None,
            )
        self._download_file(source_asset, destination)
        return AcquiredAsset(
            filename=source_asset.filename,
            source_url=source_asset.source_url,
            sha256=source_asset.expected_sha256,
            size_bytes=destination.stat().st_size,
            retrieved_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )

    def _download_file(self, source_asset: SourceAsset, destination: pathlib.Path) -> None:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False, suffix=".part") as temporary_stream:
            temporary_path = pathlib.Path(temporary_stream.name)
            try:
                with self.http_client.stream("GET", source_asset.source_url) as response:
                    response.raise_for_status()
                    total_bytes = 0
                    for byte_chunk in response.iter_bytes(chunk_size=65_536):
                        total_bytes += len(byte_chunk)
                        if total_bytes > self.maximum_bytes:
                            raise ValueError(f"Source exceeds download limit: {source_asset.filename}")
                        temporary_stream.write(byte_chunk)
                temporary_stream.flush()
                if file_checksum(temporary_path) != source_asset.expected_sha256:
                    raise ValueError(f"Downloaded checksum mismatch: {source_asset.filename}; review source version")
                temporary_path.replace(destination)
            finally:
                temporary_path.unlink(missing_ok=True)
