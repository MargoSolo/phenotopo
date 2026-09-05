import os

import pytest

from phenotopo.cli import cache_dir, cached_ontology, install_ontology, main


def test_demo_writes_a_report(tmp_path):
    out = tmp_path / "demo.html"
    main(["demo", "--no-open", "-o", str(out)])
    html = out.read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and "Phenotyping quality" in html


def test_report_from_a_csv_table(tmp_path, capsys):
    table = tmp_path / "cohort.csv"
    rows = ["id,hpo,diagnosis"]
    for i in range(40):
        terms = "HP:0001250;HP:0002133" if i % 2 else "HP:0004322;HP:0002650"
        rows.append(f"P{i},{terms},{'A' if i % 2 else 'B'}")
    table.write_text("\n".join(rows))
    obo = tmp_path / "toy.obo"
    obo.write_text("""[Term]
id: HP:0000001
name: All

[Term]
id: HP:0001250
name: Seizure
is_a: HP:0000001 ! All

[Term]
id: HP:0002133
name: Status epilepticus
is_a: HP:0001250 ! Seizure

[Term]
id: HP:0004322
name: Short stature
is_a: HP:0000001 ! All

[Term]
id: HP:0002650
name: Scoliosis
is_a: HP:0000001 ! All
""")
    out = tmp_path / "r.html"
    main(["report", str(table), "-o", str(out), "--labels", "diagnosis", "--sep", ";",
          "--ontology", str(obo), "--compare", "A", "B", "--n-perm", "50", "--no-open"])
    assert "Outliers" in out.read_text(encoding="utf-8")
    assert "40 patients" in capsys.readouterr().out


def test_ontology_cache_installs_from_a_local_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PHENOTOPO_CACHE", str(tmp_path / "cache"))
    assert cached_ontology("toy") is None
    src = tmp_path / "src.obo"
    src.write_text("[Term]\nid: HP:0000001\nname: All\n")
    path = install_ontology("toy", source=str(src))          # no network involved
    assert os.path.dirname(path) == cache_dir() and cached_ontology("toy") == path


def test_report_without_an_ontology_exits_with_advice(tmp_path, monkeypatch):
    monkeypatch.setenv("PHENOTOPO_CACHE", str(tmp_path / "empty"))
    table = tmp_path / "c.csv"
    table.write_text("id,hpo\nP1,HP:0001250\n")
    with pytest.raises(SystemExit) as exc:
        main(["report", str(table), "-o", str(tmp_path / "x.html"), "--no-open"])
    assert "ontology install" in str(exc.value)
