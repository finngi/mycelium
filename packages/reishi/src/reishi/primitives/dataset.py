"""Dataset: a versioned storage prefix plus its card and leak contract."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NotRequired, TypedDict, cast

from reishi import store


class DatasetManifest(TypedDict):
    name: str
    uri: str
    task: NotRequired[str]
    revision: NotRequired[str]
    card: NotRequired[str]
    eval_only: NotRequired[bool]
    disjoint_from: NotRequired[list[str]]


@dataclass(frozen=True)
class Dataset:
    name: str  # convention: <slug>-<ddmmyy>, e.g. identify-orgs-040726
    uri: str  # gs:// prefix (or local path during development)
    task: str = ""  # advisory provenance only; the recipe owns the task x dataset binding
    revision: str = ""  # opaque content-id naming an immutable published prefix (GCS-native versioning)
    card: str = ""
    eval_only: bool = False
    disjoint_from: tuple[str, ...] = ()  # eval sets this must never leak into

    def to_manifest(self) -> DatasetManifest:
        return {
            "name": self.name,
            "uri": self.uri,
            "task": self.task,
            "revision": self.revision,
            "card": self.card,
            "eval_only": self.eval_only,
            "disjoint_from": list(self.disjoint_from),
        }

    @classmethod
    def from_manifest(cls, m: Mapping[str, object]) -> "Dataset":
        # m is raw store.load() output, not yet trusted as DatasetManifest-shaped -- unlike
        # to_manifest() (a construction we control), this is a read boundary from disk/Postgres.
        # The cast is that trust, made explicit exactly once, rather than implicitly via Any.
        d = cast(DatasetManifest, m)
        return cls(
            name=d["name"],
            uri=d["uri"],
            task=d.get("task", ""),
            revision=d.get("revision", ""),
            card=d.get("card", ""),
            eval_only=d.get("eval_only", False),
            disjoint_from=tuple(d.get("disjoint_from", ())),
        )


def leaks(train: list["Dataset"], evals: list["Dataset"]) -> list[str]:
    """Structural (name-level) train/eval leak violations; empty list means clean.

    This proves nothing about content -- only that the declared contract is not
    obviously violated. Content-level disjointness (id/hash overlap) is a
    separate, heavier check owned by the executor at publish time.
    """
    eval_names = {d.name for d in evals}
    problems = []
    for d in train:
        if d.eval_only:
            problems.append(f"{d.name} is eval_only but used as a training input")
        if d.name in eval_names:
            problems.append(f"{d.name} is used as both a training input and an eval set")
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
