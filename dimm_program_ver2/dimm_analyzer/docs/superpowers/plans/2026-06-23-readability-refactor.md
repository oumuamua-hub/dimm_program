# DIMM Analyzer Readability Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve every observable DIMM analysis behavior while splitting the pipeline and CLI support code into focused, reviewable modules with comments at non-obvious invariants.

**Architecture:** Keep `pipeline.py` and `cli.py` as compatibility facades. Move existing function bodies verbatim into `_frame_processing.py`, `_diagnostics.py`, `_summary.py`, and `_cli_support.py`, then import/re-export their symbols from the original modules. Add characterization tests before moving code and run focused regression tests after every extraction.

**Tech Stack:** Python 3.9, NumPy, SciPy, Pydantic, Typer, pytest, Ruff

---

## File Structure

- Create `src/dimm_analyzer/_frame_processing.py`: one-frame detection, fitting, identity assignment, and tracking.
- Create `src/dimm_analyzer/_diagnostics.py`: post-fit filters and diagnostic table construction.
- Create `src/dimm_analyzer/_summary.py`: summary aggregation and result consistency validation.
- Create `src/dimm_analyzer/_cli_support.py`: output-path and batch/comparison row construction without console or analysis side effects.
- Modify `src/dimm_analyzer/pipeline.py`: retain orchestration and re-export moved symbols.
- Modify `src/dimm_analyzer/cli.py`: retain Typer entry point and monkeypatch boundaries; re-export pure helpers.
- Create `tests/test_refactor_contract.py`: lock public signatures, model fields, report schema, and compatibility exports.
- Modify existing tests only if an import path must be added to verify the new internal module; do not change expected values.

## Task 1: Lock The Existing Contract

**Files:**
- Create: `tests/test_refactor_contract.py`
- Reference: `src/dimm_analyzer/models.py`
- Reference: `src/dimm_analyzer/pipeline.py`
- Reference: `src/dimm_analyzer/report.py`

- [ ] **Step 1: Add characterization tests**

Create `tests/test_refactor_contract.py` with this content:

```python
import inspect
from dataclasses import fields

from dimm_analyzer.models import AnalysisResult, BlockResult, FitResult, FrameResult
from dimm_analyzer.pipeline import analyze_frame_sequence, analyze_ser
from dimm_analyzer.report import PER_FRAME_COLUMNS


ANALYZE_SER_PARAMETERS = (
    "input_path",
    "config",
    "output_dir",
    "start_frame",
    "end_frame",
    "max_frames",
    "preview",
    "show_progress",
    "input_path_original",
    "input_was_directory",
    "ser_selection_mode",
    "companion_files_copied",
    "output_root",
    "auto_output_name",
    "input_manifest_path",
)

ANALYZE_FRAME_SEQUENCE_PARAMETERS = (
    "frames",
    "config",
    "output_dir",
    "input_path",
    "ser_metadata",
    "dark_frame",
    "saturation_level",
    "time_source",
    "fps",
    "show_progress",
    "total_frames",
    "roi_safety_rows",
    "roi_safety_summary",
    "roi_original_size_px",
    "roi_original_fallback_size_px",
    "roi_auto_shrunk",
    "input_path_original",
    "input_path_resolved_ser",
    "input_was_directory",
    "ser_selection_mode",
    "companion_files_copied",
    "output_root",
    "auto_output_name",
    "input_manifest_path",
)

FIT_RESULT_FIELDS = (
    "success",
    "failure_reason",
    "x0_global",
    "y0_global",
    "x0_roi",
    "y0_roi",
    "background",
    "amplitude",
    "sigma_x",
    "sigma_y",
    "fwhm_x",
    "fwhm_y",
    "fwhm_mean",
    "residual_rms",
    "peak",
    "peak_raw_max",
    "peak_core_max",
    "saturated_core_pixel_count",
    "saturated_roi_pixel_count",
    "hot_pixel_or_roi_outlier",
    "flux",
    "nfev",
)


def test_pipeline_public_signatures_are_stable():
    assert tuple(inspect.signature(analyze_ser).parameters) == ANALYZE_SER_PARAMETERS
    assert (
        tuple(inspect.signature(analyze_frame_sequence).parameters)
        == ANALYZE_FRAME_SEQUENCE_PARAMETERS
    )


def test_fit_result_field_order_is_stable():
    assert tuple(field.name for field in fields(FitResult)) == FIT_RESULT_FIELDS


def test_result_models_remain_dataclasses():
    assert fields(FrameResult)
    assert fields(BlockResult)
    assert fields(AnalysisResult)


def test_per_frame_schema_matches_fit_diagnostics():
    for spot_id in (1, 2):
        assert f"peak_raw_max{spot_id}" in PER_FRAME_COLUMNS
        assert f"peak_core_max{spot_id}" in PER_FRAME_COLUMNS
        assert f"saturated_core_pixel_count{spot_id}" in PER_FRAME_COLUMNS
        assert f"saturated_roi_pixel_count{spot_id}" in PER_FRAME_COLUMNS
        assert f"hot_pixel_or_roi_outlier{spot_id}" in PER_FRAME_COLUMNS
```

- [ ] **Step 2: Run the characterization tests before refactoring**

Run:

```bash
python3 -m pytest tests/test_refactor_contract.py tests/test_gaussian_fit.py tests/test_synthetic_pipeline.py -q
```

Expected: all tests pass. A failure means the constants above do not match the current contract; correct the test constants from the authoritative source before moving code.

- [ ] **Step 3: Record the baseline suite status**

Run:

```bash
python3 -m pytest -q
python3 -m ruff check .
wc -l src/dimm_analyzer/pipeline.py src/dimm_analyzer/cli.py
```

Expected: 67 tests pass after adding four characterization tests; Ruff passes; record the exact `wc -l` values in the task notes before editing either source file.

- [ ] **Step 4: Capture deterministic output artifacts before refactoring**

Run the existing synthetic output test in a fixed temporary directory, then copy its output to a baseline directory:

```bash
test ! -e /private/tmp/dimm-refactor-baseline-20260623
python3 -m pytest tests/test_synthetic_pipeline.py::test_synthetic_pipeline_writes_outputs -q --basetemp=/private/tmp/dimm-refactor-run-20260623
cp -R /private/tmp/dimm-refactor-run-20260623/test_synthetic_pipeline_writes0 /private/tmp/dimm-refactor-baseline-20260623
find /private/tmp/dimm-refactor-baseline-20260623 -maxdepth 1 -type f -print | sort
```

Expected: the test passes and the baseline directory contains every CSV, JSON, and PNG asserted by `test_synthetic_pipeline_writes_outputs`.

- [ ] **Step 5: Commit the contract tests**

```bash
git add tests/test_refactor_contract.py
git commit -m "test: lock DIMM refactor contracts"
```

## Task 2: Extract One-Frame Processing

**Files:**
- Create: `src/dimm_analyzer/_frame_processing.py`
- Modify: `src/dimm_analyzer/pipeline.py:19-33,54-61,1373-1583`
- Test: `tests/test_synthetic_pipeline.py`
- Test: `tests/test_gaussian_fit.py`
- Test: `tests/test_refactor_contract.py`

- [ ] **Step 1: Create the frame-processing module**

Use this import and type-alias header:

```python
"""Turn one image frame into a fitted and quality-classified frame result."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .config import AnalysisConfig
from .detection import detect_sources, extract_roi, select_spot_pair
from .gaussian_fit import fit_gaussian_2d
from .models import FitResult, FrameResult
from .quality import frame_reject_reason_for_fit_fail

PreviousCenters = Tuple[float, float, float, float]
ProcessFrameReturn = Tuple[
    FrameResult,
    Optional[PreviousCenters],
    Optional[np.ndarray],
    Optional[np.ndarray],
]
```

Move these existing definitions from `pipeline.py` into the new module without changing their signatures, statements, condition order, return values, or exception handling:

```text
_process_frame
_fit_source
_large_center_jump
_assign_spot_ids
```

Add only these two comments:

```python
# Early returns below encode rejection precedence; keep their order stable.
```

Place it immediately before the first rejection branch in `_process_frame`.

```python
# Data-quality failures are final; a larger ROI cannot make them trustworthy.
```

Place it immediately before the fallback-ROI condition in `_fit_source`.

- [ ] **Step 2: Re-export the moved symbols from `pipeline.py`**

Remove the now-unused detection, Gaussian-fit, `FitResult`, and fit-failure imports from `pipeline.py`. Add:

```python
from ._frame_processing import (
    PreviousCenters,
    ProcessFrameReturn,
    _assign_spot_ids,
    _fit_source,
    _large_center_jump,
    _process_frame,
)
```

Remove the duplicate `PreviousCenters` and `ProcessFrameReturn` aliases from `pipeline.py`. Keep `FrameInput` in `pipeline.py` because it belongs to sequence orchestration.

- [ ] **Step 3: Verify frame behavior**

Run:

```bash
python3 -m pytest tests/test_gaussian_fit.py tests/test_synthetic_pipeline.py tests/test_refactor_contract.py -q
python3 -m ruff check src/dimm_analyzer/_frame_processing.py src/dimm_analyzer/pipeline.py
```

Expected: all focused tests and Ruff checks pass with unchanged expected values.

- [ ] **Step 4: Commit the frame extraction**

```bash
git add src/dimm_analyzer/_frame_processing.py src/dimm_analyzer/pipeline.py
git commit -m "refactor: isolate per-frame DIMM processing"
```

## Task 3: Extract Diagnostics And Filters

**Files:**
- Create: `src/dimm_analyzer/_diagnostics.py`
- Modify: `src/dimm_analyzer/pipeline.py:336-338,627-974,1723-1761`
- Test: `tests/test_synthetic_pipeline.py`
- Test: `tests/test_roi_safety.py`

- [ ] **Step 1: Create the diagnostics module**

Use this header:

```python
"""Post-fit filtering and diagnostic table construction."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import AnalysisConfig
from .dimm_math import compute_dimm_block
from .models import FrameResult
from .orientation import rotate_motion
from .quality import ratio
from .roi_safety import build_roi_safety_point
```

Move these constants and definitions verbatim from `pipeline.py`:

```text
REJECT_REASON_ORDER
FRAME_DIAGNOSTIC_METRICS
apply_fwhm_outlier_filter
apply_relative_motion_filter
build_spot_assignment_rows
build_roi_safety_points_from_frame_results
_frame_in_reliable_roi_population
_frame_spot_position
build_orientation_scan_rows
reject_reason_counts
build_rejection_summary_rows
build_frame_distribution_rows
_frame_metric_value
_distribution_row
_reference_series
_is_relative_outlier
_finite_or_nan
_none_if_nan
```

- [ ] **Step 2: Re-export diagnostics from `pipeline.py`**

Add this import block and remove the moved definitions plus imports that are no longer used locally:

```python
from ._diagnostics import (
    FRAME_DIAGNOSTIC_METRICS,
    REJECT_REASON_ORDER,
    _distribution_row,
    _finite_or_nan,
    _frame_in_reliable_roi_population,
    _frame_metric_value,
    _frame_spot_position,
    _is_relative_outlier,
    _none_if_nan,
    _reference_series,
    apply_fwhm_outlier_filter,
    apply_relative_motion_filter,
    build_frame_distribution_rows,
    build_orientation_scan_rows,
    build_rejection_summary_rows,
    build_roi_safety_points_from_frame_results,
    build_spot_assignment_rows,
    reject_reason_counts,
)
```

Immediately before the two filter calls in `analyze_frame_sequence`, add:

```python
# Filter order is observable: relative-motion references use the frames left by FWHM filtering.
```

- [ ] **Step 3: Verify diagnostics behavior**

Run:

```bash
python3 -m pytest tests/test_synthetic_pipeline.py tests/test_roi_safety.py tests/test_refactor_contract.py -q
python3 -m ruff check src/dimm_analyzer/_diagnostics.py src/dimm_analyzer/pipeline.py
```

Expected: all focused tests pass, including `bad_relative_motion`, hot-pixel classification, ROI safety, and output schema assertions.

- [ ] **Step 4: Commit the diagnostics extraction**

```bash
git add src/dimm_analyzer/_diagnostics.py src/dimm_analyzer/pipeline.py
git commit -m "refactor: isolate DIMM diagnostics"
```

## Task 4: Extract Summary Construction

**Files:**
- Create: `src/dimm_analyzer/_summary.py`
- Modify: `src/dimm_analyzer/pipeline.py:441-468,584-625,974-1370,1705-1722,1762-1783`
- Test: `tests/test_synthetic_pipeline.py`
- Test: `tests/test_refactor_contract.py`

- [ ] **Step 1: Create the summary module**

Use this header:

```python
"""Aggregate analysis results and validate the public summary contract."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ._diagnostics import _finite_or_nan, _frame_metric_value
from .config import AnalysisConfig, model_to_dict
from .dimm_math import seeing_from_r0_arcsec, zenith_correction_factor
from .exceptions import PipelineError
from .models import AnalysisResult, BlockResult, FrameResult, OrientationResult, SERMetadata
from .quality import ratio
```

Move these definitions verbatim from `pipeline.py`:

```text
validate_result_consistency
_saturation_summary
_valid_rejected_medians
_orientation_mismatch_score
_result_reliability
_array_stat
_valid_blocks_with_finite_seeing
_is_finite_number
_optional_close
_representative_block_values
_sanity_check_from_r0
build_summary
_estimated_fps
_nanmedian
_nanmean
_nanstd
```

Before the core/outlier count loop in `_saturation_summary`, add:

```python
# Core saturation takes precedence; edge-only pixels remain a separate diagnostic class.
```

At the start of `validate_result_consistency`, after its docstring, add:

```python
# Frame and block tables are the source of truth for duplicated summary values.
```

- [ ] **Step 2: Re-export summary symbols from `pipeline.py`**

Add:

```python
from ._summary import (
    _array_stat,
    _estimated_fps,
    _is_finite_number,
    _nanmean,
    _nanmedian,
    _nanstd,
    _optional_close,
    _orientation_mismatch_score,
    _representative_block_values,
    _result_reliability,
    _sanity_check_from_r0,
    _saturation_summary,
    _valid_blocks_with_finite_seeing,
    _valid_rejected_medians,
    build_summary,
    validate_result_consistency,
)
```

Remove imports now owned only by `_summary.py`: `model_to_dict`, `seeing_from_r0_arcsec`, and `zenith_correction_factor`. Keep `PipelineError` in `pipeline.py` because `analyze_ser` still raises it when dark subtraction is enabled without a dark path.

- [ ] **Step 3: Verify summary behavior and key order**

Run:

```bash
python3 -m pytest tests/test_synthetic_pipeline.py tests/test_refactor_contract.py -q
python3 -m ruff check src/dimm_analyzer/_summary.py src/dimm_analyzer/pipeline.py
```

Expected: summary consistency tests, null-seeing tests, saturation counts, and all existing summary key assertions pass.

- [ ] **Step 4: Commit the summary extraction**

```bash
git add src/dimm_analyzer/_summary.py src/dimm_analyzer/pipeline.py
git commit -m "refactor: isolate DIMM summary aggregation"
```

## Task 5: Extract Pure CLI Support

**Files:**
- Create: `src/dimm_analyzer/_cli_support.py`
- Modify: `src/dimm_analyzer/cli.py:5-16,40-46,370-426,580-664,742-759`
- Test: `tests/test_cli_picker.py`
- Test: `tests/test_refactor_contract.py`

- [ ] **Step 1: Create the CLI support module**

Use this header:

```python
"""Pure output-path and row-building helpers for the Typer CLI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .exceptions import DimmAnalyzerError
```

Move these definitions verbatim from `cli.py`:

```text
OutputTarget
_resolve_output_target
_resolve_batch_output_root
_resolve_collision
_next_available_output_path
_ser_output_dirs
_safe_output_stem
_batch_success_row
_batch_error_row
_batch_manifest_entry
_comparison_row
```

Before the branch sequence in `_resolve_collision`, add:

```python
# Collision precedence is part of the CLI contract: overwrite wins before run-id allocation.
```

- [ ] **Step 2: Re-export CLI support from `cli.py`**

Remove the local `OutputTarget` dataclass and moved definitions. Add:

```python
from ._cli_support import (
    OutputTarget,
    _batch_error_row,
    _batch_manifest_entry,
    _batch_success_row,
    _comparison_row,
    _next_available_output_path,
    _resolve_batch_output_root,
    _resolve_collision,
    _resolve_output_target,
    _safe_output_stem,
    _ser_output_dirs,
)
```

Delete the now-unused `json` and `dataclass` imports from `cli.py`. Keep `datetime`, `timezone`, `deepcopy`, and the report writers because the remaining CLI workflows still use them.

Keep `analyze_ser`, `select_analysis_paths`, `_run_batch_analysis`, and `_run_comparison_suite` in `cli.py`. Add this comment above the import of `analyze_ser`:

```python
# Keep this facade import: CLI tests and callers replace cli.analyze_ser at runtime.
```

- [ ] **Step 3: Verify all CLI routing and manifests**

Run:

```bash
python3 -m pytest tests/test_cli_picker.py tests/test_input_resolver.py tests/test_file_picker.py -q
python3 -m ruff check src/dimm_analyzer/_cli_support.py src/dimm_analyzer/cli.py
```

Expected: all CLI, picker, batch, comparison, collision, and manifest tests pass without changing monkeypatch targets.

- [ ] **Step 4: Commit the CLI extraction**

```bash
git add src/dimm_analyzer/_cli_support.py src/dimm_analyzer/cli.py
git commit -m "refactor: isolate pure CLI support"
```

## Task 6: Add Focused Intent Comments

**Files:**
- Modify: `src/dimm_analyzer/pipeline.py`
- Modify: `src/dimm_analyzer/roi_safety.py:237-248`
- Modify: `src/dimm_analyzer/gaussian_fit.py:277-295`
- Modify: `src/dimm_analyzer/_frame_processing.py`
- Modify: `src/dimm_analyzer/_diagnostics.py`
- Modify: `src/dimm_analyzer/_summary.py`
- Modify: `src/dimm_analyzer/_cli_support.py`

- [ ] **Step 1: Add the ROI recommendation comment**

Immediately before `recommended = recommended_from_p05` in `summarize_roi_safety`, add:

```python
# Use robust p05 for recommendations so one edge outlier does not collapse the ROI size.
```

- [ ] **Step 2: Add an orchestration comment**

Immediately before `orientation = estimate_orientation(...)` in `analyze_frame_sequence`, add:

```python
# Orientation uses only frames that survived every frame-level quality filter above.
```

- [ ] **Step 3: Explain saturation precedence at its decision point**

Immediately before the core-saturation branch in `_saturation_failure_reason`, add:

```python
# A saturated core invalidates the stellar profile; isolated ROI pixels are classified separately.
```

- [ ] **Step 4: Check comment quality**

Run:

```bash
rg -n "^\s*#" src/dimm_analyzer/pipeline.py src/dimm_analyzer/_*.py src/dimm_analyzer/roi_safety.py src/dimm_analyzer/gaussian_fit.py
```

Expected: comments explain ordering, source-of-truth, robust-statistic, compatibility, or fallback rationale; none merely restate an assignment.

- [ ] **Step 5: Run focused tests and commit comments**

```bash
python3 -m pytest tests/test_roi_safety.py tests/test_synthetic_pipeline.py tests/test_cli_picker.py -q
python3 -m ruff check .
git add src/dimm_analyzer/pipeline.py src/dimm_analyzer/roi_safety.py src/dimm_analyzer/gaussian_fit.py src/dimm_analyzer/_frame_processing.py src/dimm_analyzer/_diagnostics.py src/dimm_analyzer/_summary.py src/dimm_analyzer/_cli_support.py
git commit -m "docs: explain DIMM processing invariants"
```

## Task 7: Final Behavior And Readability Audit

**Files:**
- Verify: `README.md`
- Verify: `configs/vmc260l_imx432.yaml`
- Verify: `src/dimm_analyzer/models.py`
- Verify: `src/dimm_analyzer/report.py`
- Verify: all modified and created modules

- [ ] **Step 1: Run the complete automated suite**

```bash
python3 -m pytest
python3 -m ruff check .
python3 -m compileall -q src/dimm_analyzer
git diff --check e55aee3..HEAD
```

Expected: 67 tests pass, Ruff passes, compilation succeeds, and no whitespace errors are reported.

- [ ] **Step 2: Re-run extended maintainability checks**

```bash
python3 -m ruff check src tests --select C901,PLR0912,PLR0913,PLR0915 --output-format concise
wc -l src/dimm_analyzer/pipeline.py src/dimm_analyzer/cli.py src/dimm_analyzer/_*.py
```

Expected: existing public facade signatures may still trigger argument-count rules, but `pipeline.py` is substantially smaller and frame/diagnostic/summary logic is no longer reported as one monolithic file.

- [ ] **Step 3: Compare generated artifacts with the pre-refactor baseline**

Regenerate the same synthetic output at the same `tmp_path`, so serialized absolute paths are identical, and compare all non-image files byte-for-byte:

```bash
python3 -m pytest tests/test_synthetic_pipeline.py::test_synthetic_pipeline_writes_outputs -q --basetemp=/private/tmp/dimm-refactor-run-20260623
diff -ru --exclude='*.png' /private/tmp/dimm-refactor-baseline-20260623 /private/tmp/dimm-refactor-run-20260623/test_synthetic_pipeline_writes0
```

Expected: pytest passes and `diff` produces no output.

Compare the PNG file names, shapes, dtypes, and pixel arrays:

```bash
python3 - <<'PY'
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

baseline = Path("/private/tmp/dimm-refactor-baseline-20260623")
candidate = Path(
    "/private/tmp/dimm-refactor-run-20260623/test_synthetic_pipeline_writes0"
)
baseline_names = sorted(path.name for path in baseline.glob("*.png"))
candidate_names = sorted(path.name for path in candidate.glob("*.png"))
assert candidate_names == baseline_names
for name in baseline_names:
    before = mpimg.imread(baseline / name)
    after = mpimg.imread(candidate / name)
    assert after.shape == before.shape, name
    assert after.dtype == before.dtype, name
    assert np.array_equal(after, before), name
print(f"matched {len(baseline_names)} PNG files")
PY
```

Expected: the script prints the number of matching PNG files and raises no assertion.

- [ ] **Step 4: Audit public and output contracts mechanically**

Run:

```bash
python3 -m pytest tests/test_refactor_contract.py tests/test_synthetic_pipeline.py tests/test_cli_picker.py -q
rg -n "input_manifest.json|batch_summary.csv|batch_manifest.json|--auto-output-name|--append-run-id" README.md src/dimm_analyzer/cli.py
rg -n "peak_raw_max|peak_core_max|saturated_core_pixel_count|saturated_roi_pixel_count|hot_pixel_or_roi_outlier" src/dimm_analyzer/models.py src/dimm_analyzer/report.py src/dimm_analyzer/_summary.py
```

Expected: contract tests pass; README still documents the current CLI/output artifacts; every saturation diagnostic remains connected from model to report and summary.

- [ ] **Step 5: Inspect the final diff for accidental logic edits**

```bash
git diff --stat e55aee3..HEAD
git diff --color-moved=dimmed-zebra e55aee3..HEAD -- src/dimm_analyzer
git status --short
```

Expected: implementation changes are function moves, import rewiring, module docstrings, and focused comments. No changed numeric constants, comparison operators, rejection strings, dictionary keys, CSV columns, CLI defaults, or user-facing messages appear.

- [ ] **Step 6: Record final verification in the handoff**

Report the exact pytest count, Ruff result, remaining extended-complexity findings, before/after file sizes, and any warnings. Do not claim behavior preservation if any contract test or focused output test is skipped.
