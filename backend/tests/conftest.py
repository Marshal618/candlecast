import os
import tempfile

# Tests must never write to the real journal, sim sessions or research log; set before app.config is imported
_tmp = tempfile.mkdtemp(prefix="kd-test-")
os.environ.setdefault("KD_JOURNAL_PATH", os.path.join(_tmp, "journal.sqlite3"))
os.environ.setdefault("KD_SIM_PATH", os.path.join(_tmp, "sim.sqlite3"))
os.environ.setdefault("KD_RESEARCH_PATH", os.path.join(_tmp, "research.sqlite3"))
