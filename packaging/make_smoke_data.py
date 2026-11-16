"""Write a small synthetic survey (8 overlapping TIFFs) for testing the executable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_pipeline import _survey  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
_survey(out)
print(f"wrote {len(list(out.glob('*.tif')))} images to {out}")
