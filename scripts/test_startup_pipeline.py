from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Thread
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.startup_pipeline_state import (
    begin_run,
    mark_failed,
    mark_ready,
    wait_until_ready,
)


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        state_dir = Path(temp_dir)
        run_id_file = state_dir / "run-id"
        ready_file = state_dir / "docs-ready"
        failed_file = state_dir / "docs-failed"

        first_run = begin_run(run_id_file, ready_file, failed_file)
        mark_ready(run_id_file, ready_file, failed_file)
        if wait_until_ready(
            run_id_file,
            ready_file,
            timeout_seconds=0.1,
            failed_file=failed_file,
        ) != first_run:
            raise AssertionError("Ready marker should match the active startup run.")

        second_run = begin_run(run_id_file, ready_file, failed_file)
        if second_run == first_run or ready_file.exists() or failed_file.exists():
            raise AssertionError("A new run must invalidate old terminal markers.")

        mark_failed(run_id_file, failed_file)
        try:
            wait_until_ready(
                run_id_file,
                ready_file,
                timeout_seconds=1.0,
                failed_file=failed_file,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("A failed run should stop waiters immediately.")

        mark_ready(run_id_file, ready_file, failed_file)
        if failed_file.exists():
            raise AssertionError("A recovered run must clear its failure marker.")

        third_run = begin_run(run_id_file, ready_file, failed_file)

        def delayed_ready() -> None:
            time.sleep(0.05)
            mark_ready(run_id_file, ready_file, failed_file)

        worker = Thread(target=delayed_ready)
        worker.start()
        observed = wait_until_ready(
            run_id_file,
            ready_file,
            timeout_seconds=1.0,
            failed_file=failed_file,
        )
        worker.join()
        if observed != third_run:
            raise AssertionError("Waiter should unblock only for the current run id.")

    print("Startup pipeline state handoff -> ok")


if __name__ == "__main__":
    main()
