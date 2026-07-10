"""Dataset: a name, a storage location, and leak metadata, round-tripped
through a manifest. leaks() does name-level train/eval overlap checks.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NotRequired, TypedDict, cast

from reishi import store


class DatasetManifest(TypedDict):
    name: str
    uri: str
    advisory_task: NotRequired[str]
    revision: NotRequired[str]
    card: NotRequired[str]
    eval_only: NotRequired[bool]
    disjoint_from: NotRequired[list[str]]


@dataclass(frozen=True)
class Dataset:
    name: str  # the key this dataset is saved and loaded under
    uri: str  # opaque location string; carried into the manifest, never read here
    advisory_task: str = ""  # hints the task; never binds it
    revision: str = ""  # opaque version id, carried in the manifest
    card: str = ""
    eval_only: bool = False
    disjoint_from: tuple[str, ...] = ()  # eval-set names leaks() flags on overlap

    def to_manifest(self) -> DatasetManifest:
        return {
            "name": self.name,
            "uri": self.uri,
            "advisory_task": self.advisory_task,
            "revision": self.revision,
            "card": self.card,
            "eval_only": self.eval_only,
            "disjoint_from": list(self.disjoint_from),
        }

    @classmethod
    def from_manifest(cls, m: Mapping[str, object]) -> "Dataset":
        # store.load() returns an untyped Mapping; cast it to the manifest shape
        # at this read boundary rather than threading Any through the reads below.
        d = cast(DatasetManifest, m)
        return cls(
            name=d["name"],
            uri=d["uri"],
            advisory_task=d.get("advisory_task", ""),
            revision=d.get("revision", ""),
            card=d.get("card", ""),
            eval_only=d.get("eval_only", False),
            disjoint_from=tuple(d.get("disjoint_from", ())),
        )


def leaks(train: list["Dataset"], evals: list["Dataset"]) -> list[str]:
    """Name-level train/eval overlap violations; empty list means none found.

    Compares declared names only -- does not inspect dataset contents.
    """
    eval_names = {d.name for d in evals}
    problems = []
    for d in train:
        if d.eval_only:
            problems.append(f"{d.name} is eval_only but used as a training input")
        if d.name in eval_names:
            problems.append(
                f"{d.name} is used as both a training input and an eval set"
            )
        for forbidden in d.disjoint_from:
            if forbidden in eval_names:
                problems.append(
                    f"{d.name} declares disjoint_from '{forbidden}', but that eval set is in this run"
                )
    return problems


def save(ds: Dataset) -> None:
    store.save("datasets", ds.name, ds.to_manifest())


def load(name: str) -> Dataset:
    return Dataset.from_manifest(store.load("datasets", name))


def load_all() -> list[Dataset]:
    return [Dataset.from_manifest(m) for m in store.load_all("datasets")]
