"""Dataset: a versioned storage prefix plus its card and leak contract."""

from dataclasses import dataclass, field

from mcm import store


@dataclass(frozen=True)
class Dataset:
    name: str  # convention: <slug>-<ddmmyy>, e.g. identify-orgs-040726
    uri: str  # gs:// prefix (or local path during development)
    task: str
    card: str = ""
    eval_only: bool = False
    disjoint_from: tuple[str, ...] = ()  # eval sets this must never leak into

    def to_manifest(self) -> dict:
        return {
            "name": self.name,
            "uri": self.uri,
            "task": self.task,
            "card": self.card,
            "eval_only": self.eval_only,
            "disjoint_from": list(self.disjoint_from),
        }

    @classmethod
    def from_manifest(cls, m: dict) -> "Dataset":
        return cls(
            name=m["name"],
            uri=m["uri"],
            task=m["task"],
            card=m.get("card", ""),
            eval_only=m.get("eval_only", False),
            disjoint_from=tuple(m.get("disjoint_from", ())),
        )


def save(ds: Dataset) -> None:
    store.save("datasets", ds.name, ds.to_manifest())


def load(name: str) -> Dataset:
    return Dataset.from_manifest(store.load("datasets", name))


def load_all() -> list[Dataset]:
    return [Dataset.from_manifest(m) for m in store.load_all("datasets")]
