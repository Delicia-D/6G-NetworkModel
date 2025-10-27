## Test Suite

Run each test directly from the repository root.  
On Windows, file names are case-insensitive; on macOS/Linux they are case-sensitive — use the exact casing shown.

| Script                         | What it tests                                      | Run command                           |
| ------------------------------ | -------------------------------------------------- | ------------------------------------- |
| `Testmain.py`                  | End-to-end main simulation (default settings)      | `python Testmain.py`                  |
| `Testservicetypes.py`          | Voice vs. video RB usage / service mix impact      | `python Testservicetypes.py`          |
| `TestvaryingDuration.py`       | Effect of different callduration distribution         | `python TestvaryingDuration.py`       |
| `Testvaryinggroups.py`         | Coverage groups A–D composition                       | `python Testvaryinggroups.py`         |
| `Testvaryingheight.py`         | Satellite altitude / visibility sensitivity        | `python Testvaryingheight.py`         |
| `TestVaryingPredictability.py` | Impact of predictor accuracy on CAC decisions      | `python TestVaryingPredictability.py` |

**Quick start (main run):**
```bash
python Testmain.py

