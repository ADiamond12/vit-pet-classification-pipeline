from pathlib import Path


def test_streamlit_ui_keeps_public_boundary_language():
    ui_source = Path("src/ui/app.py").read_text(encoding="utf-8")

    assert "Public-safe Streamlit review UI" in ui_source
    assert "does not ship model weights or datasets" in ui_source
    assert "project-owned" in ui_source
    assert "checkpoint" in ui_source
    assert "Prediction remains inactive" in ui_source
    assert "What this public UI proves" in ui_source
    assert "Backend readiness is explicit" in ui_source
