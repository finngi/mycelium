from reishi.primitives.task import Task, register
from reishi.tasks.nameparse.scoring import score

# Schema matches the IF name-parser PascalCase output; the field-F1 scorer
# ports from mycelium's eval_common (see scoring.py). Constrained decoding
# doesn't port yet -- mlx_lm 0.31.3 has no structured-decoding API, so MLX
# trials fall back to raw generation + tolerant codec decode.
nameparse = register(
    Task(
        name="nameparse",
        description="string -> structured person/org name parse (IF name-parser distillation)",
        output_fields=(
            "FirstName",
            "FamilyName",
            "FirstNameShortestDiminutives",
            "DistinctiveTitles",
            "Script",
            "TopLevelBrand",
        ),
        score=score,
    )
)
