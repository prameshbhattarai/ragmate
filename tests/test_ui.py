from pathlib import Path

from streamlit.testing.v1 import AppTest

UI = str(Path(__file__).resolve().parent.parent / "ragmate" / "ui.py")


def test_ui_renders_on_an_empty_store_without_calling_a_model():
    at = AppTest.from_file(UI, default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "📚 ragmate"
    assert any("store is empty" in info.value for info in at.info)
    assert [r.value for r in at.radio] == ["agent"]
