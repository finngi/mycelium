from mcm.primitives.task import Task, register

# Schema matches the IF name-parser PascalCase output; the field-F1 scorer and
# constrained decoder port from mycelium (eval_common / constrained_decoding)
# when the first trainer adapter lands.
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
        score=None,
    )
)
