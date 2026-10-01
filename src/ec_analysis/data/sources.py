"""Official, pinned Release 47 format fixtures; not the annual research sample."""

import dataclasses
import typing


@typing.final
@dataclasses.dataclass(kw_only=True, frozen=True, slots=True)
class SourceAsset:
    filename: str
    source_url: str
    expected_sha256: str


RELEASE47_ASSETS: typing.Final = (
    SourceAsset(
        filename="release-47-sample-files.zip",
        source_url="https://www.freddiemac.com/fmac-resources/research/docs/release-47-sample-files.zip",
        expected_sha256="c1374946c9414c8182f2f7c637776808b5c109a43d19a4ca928f9a1a4933fa06",
    ),
    SourceAsset(
        filename="file_layout_july_2026.xlsx",
        source_url="https://www.freddiemac.com/fmac-resources/research/pdf/file_layout_july_2026.xlsx",
        expected_sha256="ce054271c42b7ad5f173a045c73368d997a2ac99253dcb312a45ccef43a4b13e",
    ),
    SourceAsset(
        filename="general_user_guide_july_2026.pdf",
        source_url="https://www.freddiemac.com/fmac-resources/research/pdf/general_user_guide_july_2026.pdf",
        expected_sha256="4693ae754109c213feb84e97702fa675ed3926c2a5f4b6f7ba3c347db2aaf03e",
    ),
)
