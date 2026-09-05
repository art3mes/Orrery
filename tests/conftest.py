import sys
from pathlib import Path

# Let the suite run straight from a checkout, installed or not.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
