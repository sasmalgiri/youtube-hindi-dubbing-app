"""T08: the server process is killed (no clean shutdown) while a paid job runs.
The next process finds the job still 'running' in the queue, requeues it,
runs it once more and finishes; a job killed on its last allowed attempt is
given up (its time released), never left hanging."""
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import workqueue

CHILD = textwrap.dedent('''
    import sys, time
    from pathlib import Path
    sys.path.insert(0, {backend!r})
    import workqueue
    db, marker = Path({db!r}), Path({marker!r})
    def run(job_id):
        marker.write_text(job_id)
        time.sleep(120)            # killed long before this ends
    q = workqueue.WorkQueue(db, run, max_global=2, max_per_account=2)
    for j in {jobs!r}:
        q.enqueue(j, "acct")
    q.start()
    time.sleep(120)
''')


def _crash_while_running(tmp_path, jobs, attempts_before=0):
    db, marker = tmp_path / "queue.db", tmp_path / "started"
    if attempts_before:
        q = workqueue.WorkQueue(db, lambda j: None)
        for j in jobs:
            q.enqueue(j, "acct")
            q._x("UPDATE work SET attempts=? WHERE job_id=?", (attempts_before, j))
        q.close()
    code = CHILD.format(backend=str(Path(workqueue.__file__).parent), db=str(db), marker=str(marker), jobs=jobs)
    p = subprocess.Popen([sys.executable, "-c", code])
    try:
        for _ in range(200):
            if marker.exists():
                break
            time.sleep(0.05)
        assert marker.exists(), "child never started the job"
    finally:
        p.kill()                    # hard kill: no stop(), no lease release
        p.wait(10)
    return db


def test_killed_mid_job_is_resumed_once(tmp_path):
    db = _crash_while_running(tmp_path, ["j1"])
    ran, gave_up = [], []
    q = workqueue.WorkQueue(db, ran.append, on_give_up=lambda j, why: gave_up.append(j), max_global=2, max_per_account=2)
    assert q.row("j1")["state"] == "running"          # what the dead process left behind
    assert q.recover() == ["j1"]
    q.start()
    assert q.wait_idle(20)
    q.close()
    assert ran == ["j1"] and gave_up == []


def test_killed_on_last_attempt_is_given_up_not_stuck(tmp_path):
    db = _crash_while_running(tmp_path, ["j2"], attempts_before=1)   # child run = attempt 2 of 2
    ran, gave_up = [], []
    q = workqueue.WorkQueue(db, ran.append, on_give_up=lambda j, why: gave_up.append((j, why)), max_global=2, max_per_account=2)
    q.recover()
    q.start()
    q.wait_idle(10)
    q.close()
    assert ran == [] and [g[0] for g in gave_up] == ["j2"]
