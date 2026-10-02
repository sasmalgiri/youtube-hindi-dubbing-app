import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

# app.py works under VOICEDUB_WORK (default C:/tmp/vd) and, whenever a job is
# created (the API tests create several), deletes every outputs/ folder older
# than 2 h that is not in ITS jobs.db -- from a worktree that is every real
# job's folder. Tests get a throwaway work root (set before app is imported).
_WORK = tempfile.mkdtemp(prefix="voicedub-tests-")
os.environ["VOICEDUB_WORK"] = _WORK
# ...and its own jobs.db / saved_links.json / completed_urls.json: the API tests
# create jobs, which used to land in the real job history (32 ghost jobs found).
os.environ["VOICEDUB_STATE_DIR"] = _WORK
atexit.register(shutil.rmtree, _WORK, True)
