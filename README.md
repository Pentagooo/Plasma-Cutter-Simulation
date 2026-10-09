# Plasma Cutter — Path Planner & Segment Simulation

Path planner for a robot-guided plasma cutter. From a part geometry (outer
contour + holes, given as a point grid), the system computes a **complete,
time-optimal and collision-free cutting primitive**: which pieces of the
contour are cut, in which order and direction, and at which speed?

## Model

- The TCP keeps a constant minimum gap `d_min` to the material while cutting
- The length of the plasma jet decreases linearly with speed:
  `L(v) = L0 − slope · v`
- **Coverage** = fraction of grid points reached by the jet
- **Goal**: full coverage in minimum execution time `T`, within the planning
  budget

| Quantity | Meaning | Code |
|---|---|---|
| Blade length `L(v)` | reach of the jet; faster = shorter | `cutter/assumptions.py` (`BladeLengthModel`), `Cutter.blade_length(v)` |
| Effective depth | `L(v) − d_min` | `planning.RunKinematics.effective_depth` |
| Swept Area | area swept by the jet along a cut | `RunKinematics.attach` (Shapely polygon per `CutRun`) |
| Coverage | fraction of grid points inside the swept areas | `segments.compute_grid_coverage` |
| Pierce time | flat time per pierce; skipped if the adjacent segment was cut directly before | `Cutter.pierce_time()`, `PierceTimeModel` |
| Speed rule | each section of a cut runs at the highest speed at which the jet still reaches the deepest point to be cut; each speed change costs `t_switch` | `surrogate/runutils.py` (`split_group_for_speed`, `build_speed_chains`) |
| `T` | `T_cut + T_rapid + T_pierce + T_switch` | `runutils.build_plan_with_speeds` → `SpeedPlan.total_time` |

## Pipeline

All methods run through the same pipeline and differ only in the selection:

```
PointGrid
  -> SegmentedContour.from_grid   split the contour into primitive segments
  -> selection                    manual / Automatic Planner / Surrogate / Brute Force
  -> build_speed_chains           adjacent segments into chains, DP split into
                                  speed blocks, verify each block exactly
  -> build_plan_with_speeds       Sequencer: LinkPlanner (collision-free transitions,
                                  t_link + pierce) -> Held-Karp (order, direction),
                                  then time breakdown T
  -> compute_grid_coverage        exact coverage
```

| Method | Selection | Entry point |
|---|---|---|
| **Manual** | pick start and end node of each cut by mouse click | simulator (click) |
| **Automatic Planner** | Greedy Set Cover + pruning of the `AutoPlanner` | key P, `surrogate.planner.greedy_plus_plan` |
| **Surrogate** | a learned model ranks the segments by p(s) "is in the optimum"; Greedy Set Cover in this order on exact masks, pruning, verify, fallback to greedy; coverage never depends on the model | key S, `surrogate.planner.surrogate_plan` |
| **Brute Force** | all 2^k segment subsets, every complete cover evaluated exactly = optimum (up to `K_MAX_DEFAULT` segments); also produces the training labels | key B, `surrogate.teacher.exhaustive_plan` |

## Quick start

- Python ≥ 3.10 (tested with 3.14)
- The folder must be named `plasma_cutter`; run commands from its parent folder
- Model saved with scikit-learn 1.9; the `params_hash` warning on loading is
  expected (the model comes from an older catalog version)

```bash
git clone https://igm-git.igm.rwth-aachen.de/sherec/auto_cutting_primitives.git plasma_cutter
cd plasma_cutter
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
python -m pip install -r requirements.txt
cd ..
```

### Interactive simulator

```bash
python -m plasma_cutter.segment_simulation.simulation \
    --geometry "plasma_cutter/geometry/Geometrie_Konturen_geprüft/kontur.json"
# without --geometry: file dialog; parameters: --v-cut --v-max --blade-length --blade-slope --clearance --segment-length
```

| Key / button | Action |
|---|---|
| click | manual selection: start node, then end node of a cut |
| `A` | select all remaining contours completely |
| `P` / Automatic planner | selection of the Automatic Planner |
| `S` / Surrogate | surrogate selection (loads `surrogate/artifacts/surrogate_model.joblib`) |
| `B` / Brute force | exact optimum (computation time grows exponentially with the number of segments) |
| `V` / Speed rule | speed rule on/off |
| `Enter` | plan (order, links, speeds) + animate |
| `U` / right click, `R`, `Esc`, `+`/`-` | undo, reset, cancel, animation speed |

The status bar reports cuts, `T`, coverage and planning time for each method;
Enter reproduces the reported `T` exactly.

### Headless

```python
from plasma_cutter.geometry.point_grid import PointGrid
from plasma_cutter.segment_simulation.surrogate import planner, teacher, model

grid = PointGrid.from_json("plasma_cutter/geometry/Geometrie_Konturen_geprüft/kontur.json")
gp   = planner.greedy_plus_plan(grid)                 # dict: T, coverage, n_runs, t_plan
sur  = planner.surrogate_plan(grid, model.load_model())
print(sur.summary(), sur.selected)
opt  = teacher.exhaustive_plan(grid, k_max=18)        # TeacherSkipped for > k_max segments
```

## Training, benchmark, tests

```bash
# show the parameter stamp (all label-relevant values + hashes)
python -m plasma_cutter.segment_simulation.surrogate.params

# check segment counts per family (without the teacher)
python -m plasma_cutter.segment_simulation.surrogate.dataset --dry-run --n 400 [--seg-divisor 18 --seg-min-spacings 3]

# labels from the brute-force teacher (resumable, cached per instance in <out>/labels/);
# --seg-mix gives each instance a segmentation from a weighted mixture
python -m plasma_cutter.segment_simulation.surrogate.dataset --n 4200 --seed 42 --n-jobs 8 --k-max 20 \
    --seg-mix "12/4:0.50,14/4:0.25,16/3:0.20,18/3:0.05" \
    --out plasma_cutter/segment_simulation/surrogate/artifacts/runs/main [--max-minutes 600 --stop-file STOP]

# train the model  ->  <out>/surrogate_model.joblib + model_meta.json
# after changing the feature list: keep the labels, recompute X only (no relabeling)
python -m plasma_cutter.segment_simulation.surrogate.dataset --refeaturize .../runs/main/labels
python -m plasma_cutter.segment_simulation.surrogate.model --train --out .../runs/main

# label the optimum of the test set in parallel, then benchmark Automatic Planner / surrogate / Brute Force
python -m plasma_cutter.segment_simulation.surrogate.dataset --n 60 --seed 7 --include-real --n-jobs 8 --out .../runs/main_eval
python -m plasma_cutter.segment_simulation.surrogate.benchmark --n 60 --seed 7 --reps 3 \
    --model .../runs/main/surrogate_model.joblib --out .../runs/main --opt-labels .../runs/main_eval/labels

# learning curve (models on the first N instances, fixed test set seed 7)
python -m plasma_cutter.segment_simulation.surrogate.learning_curve --dataset .../runs/main \
    --n-list 100,500,2000,4000 --n-eval 60 --opt-labels .../runs/main_eval/labels

# tests (first line fast; second also runs the slow dominance tests of the teacher)
python -m pytest plasma_cutter/segment_simulation/surrogate/tests -q -m "not slow"
python -m pytest plasma_cutter/segment_simulation/surrogate/tests -q
```

## Project structure

```
plasma_cutter/
├── segment_simulation/
│   ├── segments.py            split the contour into segments, describe cuts, count coverage
│   ├── planning.py            jet and swept area per cut, collision-free traverse paths,
│   │                          best order of the cuts
│   ├── autoplan.py            Automatic Planner: picks segments until everything is covered
│   ├── simulation.py          interactive simulator with animation; physics constants of the cutter
│   └── surrogate/
│       ├── planner.py         surrogate planner and Automatic Planner with speed rule
│       ├── teacher.py         Brute Force: tries all segment combinations, returns the optimum
│       ├── model.py           train and load the learned model
│       ├── features.py        per-segment features the model learns from
│       ├── dataset.py         generate training data (Brute Force on many parts)
│       ├── instances.py       part catalog for training: standard sections, attachments,
│       │                      welded assemblies
│       ├── params.py          stamp: detects whether model, data and physics match
│       ├── runutils.py        steps shared by all planners: join segments into cuts,
│       │                      speeds, order, total time
│       ├── benchmark.py       compare the planners on unseen parts
│       ├── learning_curve.py  model quality over the amount of training data
│       ├── tests/             automated tests
│       └── artifacts/         trained model (in the repo); training runs local only
├── cutter/                    physics model of the cutter: jet length L(v), pierce time
├── geometry/                  load point grids, prepare raw contours (geometry_processor.py)
│   ├── Geometrie_Konturen_ungeprüft/   raw contours from the contour editor
│   └── Geometrie_Konturen_geprüft/     prepared test geometries
├── requirements.txt           required Python packages
└── README.md
```

## Parameters and stamp

Values that determine training data and model:

- physics of the cutter (`v_cut`, `v_max`, `L0`, `blade_slope`,
  `minimum_gap` = `d_min`, `rapid_speed`, `t_switch`, pierce):
  constants block in `segment_simulation/simulation.py`
- kerf and segmentation: `surrogate/params.py`
- sampling of the TCP path: `planning.TCP_SAMPLE_STEP`
- point density and part catalog: `surrogate/instances.py`

The stamp (`python -m plasma_cutter.segment_simulation.surrogate.params`) is
stored in every label and in the model and detects changes:

- different physics (`phys_hash`): training and loading the model abort →
  relabel and retrain
- different segmentation or catalog (`params_hash`): warning only

Add new part families to `instances.py` only by appending them, and increase
`SHAPE_VERSION`.

## Geometries

`geometry/geometry_processor.py` converts a raw contour from the contour
editor (examples in `Geometrie_Konturen_ungeprüft/`) into a checked geometry
(densified outer/hole contour + inner grid, written to
`Geometrie_Konturen_geprüft/`):

```bash
python plasma_cutter/geometry/geometry_processor.py "path/to/kontur.json" --no-plot
```
