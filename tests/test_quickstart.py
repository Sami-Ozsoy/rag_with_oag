from pathlib import Path

from rag_with_oag import Ontology

QUICKSTART = Path(__file__).resolve().parents[1] / "examples" / "quickstart"


def test_quickstart_files_and_ontology() -> None:
    ontology = Ontology.from_yaml(QUICKSTART / "ontology.yaml")

    assert ontology.name == "contracts"
    assert {entity.name: entity.key for entity in ontology.entities} == {
        "Sirket": "vergi_no",
        "Sozlesme": "numara",
    }
    assert {relation.name for relation in ontology.relations} == {"TARAFIDIR", "ORTAGIDIR"}
    for file in ("run.py", "docs/contract.md", "docs/shareholders.md", ".gitignore"):
        assert (QUICKSTART / file).is_file()
