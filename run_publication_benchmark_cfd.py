"""Run the existing publication benchmark with an explicitly required CFD field."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
from run_publication_benchmark import main

if __name__ == '__main__':
    if '--cfd-file' not in sys.argv and not any(arg.startswith('--cfd-file=') for arg in sys.argv):
        raise SystemExit('--cfd-file is required; import a SimScale ZIP with scripts/import_simscale_flow.py first')
    main()
