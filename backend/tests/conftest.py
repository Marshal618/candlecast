import os
import tempfile

# Tests must never write to the real forecast journal; set before app.config is imported
os.environ.setdefault("KD_JOURNAL_PATH", os.path.join(tempfile.mkdtemp(prefix="kd-test-"), "journal.sqlite3"))
