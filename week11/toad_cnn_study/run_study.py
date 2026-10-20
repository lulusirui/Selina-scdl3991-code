from pathlib import Path

from toad_study import StudyConfig, run_study


if __name__ == "__main__":
    # One definitive design: 50,000 fixed training pairs made from two
    # disjoint banks of 50,000 prior-predictive datasets.
    run_study(Path(__file__).resolve().parent, StudyConfig())
