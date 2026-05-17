"""Phase 4 -- Sanity-Checks fuer das RL-Environment.

Diese Datei ueberprueft, ob die Gym-Environment mechanisch fehlerfrei
ist und das Reward-Accounting mit Phase 1 (Baseline) uebereinstimmt.
Sie laeuft rein mit numpy und der Env (keine torch/SB3-Abhaengigkeit).
"""

import json
import time
import traceback
from pathlib import Path

import numpy as np

# Um relative Imports zu erlauben, wenn das Script als Modul aufgerufen wird
try:
    from .config import RLConfig, default_config
    from .env import PlasmaCutterEnv
    from .baseline import list_geometries, split_train_test, build_outer_contour_waypoints, _evaluate_geometry
    from .mdp import compute_terminal_reward
    from ..geometry.point_grid import PointGrid
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from plasma_cutter.rl.config import RLConfig, default_config
    from plasma_cutter.rl.env import PlasmaCutterEnv
    from plasma_cutter.rl.baseline import list_geometries, split_train_test, build_outer_contour_waypoints, _evaluate_geometry
    from plasma_cutter.rl.mdp import compute_terminal_reward
    from plasma_cutter.geometry.point_grid import PointGrid


def test_random_policy_smoke(cfg: RLConfig, n_episodes: int = 100) -> dict:
    env = PlasmaCutterEnv(cfg, mode="train")
    
    returns = []
    coverages = []
    steps_list = []
    termination_reasons = {"infeasible": 0, "success": 0, "timeout": 0, "other": 0}
    
    rng = np.random.default_rng(cfg.seed)
    passed = True
    
    try:
        for _ in range(n_episodes):
            obs, info = env.reset(seed=int(rng.integers(0, 1000000)))
            done = False
            ep_return = 0.0
            steps = 0
            
            while not done:
                # Random action in [-1, 1]
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                steps += 1
                
                if np.isnan(obs).any() or np.isinf(obs).any() or np.isnan(action).any() or np.isnan(reward):
                    passed = False
                    
                if not (np.all(obs >= 0.0) and np.all(obs <= cfg.observation.clip_max)):
                    passed = False
                
                done = terminated or truncated
                
            returns.append(ep_return)
            coverages.append(info["coverage"])
            steps_list.append(steps)
            
            if terminated and not info.get("is_feasible", True):
                termination_reasons["infeasible"] += 1
            elif info["coverage"] >= cfg.objective.coverage_threshold:
                termination_reasons["success"] += 1
            elif truncated:
                termination_reasons["timeout"] += 1
            else:
                termination_reasons["other"] += 1
                
    except Exception as e:
        print(f"Exception in test_random_policy_smoke: {e}")
        traceback.print_exc()
        passed = False
        
    return_var = np.var(returns) if returns else 0.0
    cov_max = np.max(coverages) if coverages else 0.0
    infeasible_rate = termination_reasons["infeasible"] / n_episodes if n_episodes else 0.0
    
    passed = passed and (return_var > 0) and (cov_max > 0) and (infeasible_rate < 0.9)
    
    return {
        "passed": bool(passed),
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
        "coverage_mean": float(np.mean(coverages)) if coverages else 0.0,
        "coverage_max": float(cov_max),
        "avg_steps": float(np.mean(steps_list)) if steps_list else 0.0,
        "termination_reasons": termination_reasons,
        "infeasible_rate": float(infeasible_rate)
    }


def test_baseline_replay(cfg: RLConfig) -> dict:
    import copy
    cfg_test = copy.deepcopy(cfg)
    cfg_test.scope.randomize_scale = False
    cfg_test.scope.randomize_rotation = False
    # Volle Trajektorie zulassen: ohne diesen Override wuerde die Env
    # mitten im Baseline-Pfad bei coverage>=0.99 terminieren und der
    # Reward-Vergleich gegen den vollstaendigen Baseline-Pfad waere
    # systematisch zu hoch.
    cfg_test.objective.coverage_threshold = 1.01
    # Phase-1-`calculate_reward` vergibt den Completion-Bonus bei
    # coverage>=0.995 statt 0.99 -- gleichziehen, damit beide Formeln
    # mathematisch identisch sind.
    cfg_test.reward.completion_threshold = 0.995
    # Der Replay zerlegt jede Baseline-Kante in mehrere Env-Steps; das
    # Step-Budget muss entsprechend grosszuegig sein, sonst truncatet
    # der Test bevor die Trajektorie fertig ist.
    cfg_test.constraints.max_steps_per_episode = 5000
    
    files = list_geometries(cfg_test)
    train_files, _ = split_train_test(files, cfg_test)
    
    results = {}
    all_passed = True
    
    for f in train_files:
        try:
            grid = PointGrid.from_json(f)
            waypoints = build_outer_contour_waypoints(grid, cfg_test.cutter.minimum_gap)
            if len(waypoints) < 2:
                continue
                
            env_test = PlasmaCutterEnv(cfg_test, mode="train")
            obs, info = env_test.reset(options={"geometry_file": f.name})
            
            # Override initial TCP to avoid rapid penalty at start
            env_test._tcp = waypoints[0].copy()
            max_step = info["max_step_mm"]
            
            total_env_reward = 0.0
            done = False
            
            for wp in waypoints[1:]:
                current_tcp = env_test._tcp
                target_tcp = wp
                
                dist = np.linalg.norm(target_tcp - current_tcp)
                
                num_steps = int(np.ceil(dist / max_step))
                if num_steps == 0:
                    continue
                    
                for i in range(1, num_steps + 1):
                    interp_wp = current_tcp + (target_tcp - current_tcp) * (i / num_steps)
                    delta = interp_wp - env_test._tcp
                    
                    action_xy = delta / max_step
                    action = np.zeros(env_test.action_space.shape, dtype=np.float32)
                    action[0] = action_xy[0]
                    action[1] = action_xy[1]
                    if cfg_test.action.include_cut_flag:
                        action[2] = 1.0 # force cut
                    
                    obs, reward, terminated, truncated, info = env_test.step(action)
                    total_env_reward += reward
                    if terminated or truncated:
                        done = True
                        break
                if done:
                    break
                    
            # Terminal-Reward manuell addieren: weil wir
            # coverage_threshold > 1 setzen, terminiert die Env nicht
            # selbst und vergibt den Completion-Bonus nicht. Phase-1-
            # `calculate_reward` vergibt ihn aber unbedingt am Endzustand.
            # Wir holen das hier nach, damit beide Formeln vergleichbar sind.
            if not done:
                term_r = compute_terminal_reward(
                    final_coverage = info["coverage"],
                    n_pierces      = info["n_pierces"],
                    is_feasible    = info.get("is_feasible", True),
                    timed_out      = False,
                    cfg            = cfg_test,
                )
                total_env_reward += term_r

            baseline_res = _evaluate_geometry(f, cfg_test)
            expected_reward = baseline_res.reward
            
            # Toleranz wegen Interpolations-Unterschieden: das Replay
            # zerlegt jede Baseline-Kante in mehrere Env-Steps, was
            # marginal mehr Verfahrzeit (Mikro-Pierce-Boundaries) erzeugt.
            diff = abs(total_env_reward - expected_reward)
            passed = diff < 5.0
            all_passed = all_passed and passed
            
            results[f.name] = {
                "env_reward": float(total_env_reward),
                "expected_reward": float(expected_reward),
                "diff": float(diff),
                "passed": bool(passed)
            }
        except Exception as e:
            print(f"Exception in test_baseline_replay for {f.name}: {e}")
            all_passed = False
            results[f.name] = {"passed": False, "error": str(e)}
        
    return {
        "passed": bool(all_passed),
        "details": results
    }


def test_reward_decomposition(cfg: RLConfig) -> dict:
    env = PlasmaCutterEnv(cfg, mode="train")
    
    passed = True
    details = []
    
    for seed in range(42, 52):
        obs, info = env.reset(seed=seed)
        done = False
        total_env_reward = 0.0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_env_reward += reward
            done = terminated or truncated
            
        # Dichter Shaping-Anteil: er teleskopiert ueber die Episode zu
        #     w_cov * cov_final^2  -  w_time * total_time
        # Das gilt UNABHAENGIG davon, ob die Episode regulaer endet oder
        # mit einem Infeasible-Step abgebrochen wird -- die Env behaelt
        # in beiden Faellen den step_reward des letzten Steps.
        dense = (
            cfg.reward.w_coverage_delta * (info["coverage"] ** 2)
            - cfg.reward.w_time_delta   * info["total_time"]
        )

        if not info.get("is_feasible", True):
            # Terminal-Pfad bei infeasible: nur die Strafe, keine
            # Completion- oder Pierce-Terme (vgl. compute_terminal_reward).
            expected_reward = dense + cfg.constraints.infeasible_penalty
        else:
            expected_reward = dense
            if info["coverage"] >= cfg.reward.completion_threshold:
                expected_reward += cfg.reward.w_completion
            expected_reward -= cfg.reward.w_pierce * max(0, info["n_pierces"] - 1)

            if truncated and info["coverage"] < cfg.reward.completion_threshold:
                expected_reward -= cfg.reward.w_failure_penalty * (1.0 - info["coverage"])
                
        diff = abs(total_env_reward - expected_reward)
        is_ok = diff < 1e-3
        passed = passed and is_ok
        details.append({
            "seed": seed,
            "env_reward": total_env_reward,
            "expected_reward": expected_reward,
            "diff": diff,
            "passed": is_ok
        })
        
    return {
        "passed": bool(passed),
        "details": details
    }


def test_determinism(cfg: RLConfig) -> dict:
    env1 = PlasmaCutterEnv(cfg, mode="train")
    env2 = PlasmaCutterEnv(cfg, mode="train")
    
    np.random.seed(42)
    actions = [np.random.uniform(-1, 1, env1.action_space.shape) for _ in range(50)]
    
    obs1, info1 = env1.reset(seed=42)
    obs2, info2 = env2.reset(seed=42)
    
    passed = True
    
    if not np.allclose(obs1, obs2): 
        passed = False
    
    for action in actions:
        o1, r1, term1, trunc1, i1 = env1.step(action)
        o2, r2, term2, trunc2, i2 = env2.step(action)
        
        if not np.allclose(o1, o2): passed = False
        if abs(r1 - r2) > 1e-6: passed = False
        if term1 != term2 or trunc1 != trunc2: passed = False
        
        if term1 or trunc1:
            break
            
    return {
        "passed": bool(passed)
    }


def test_env_speed(cfg: RLConfig, n_steps: int = 2000) -> dict:
    env = PlasmaCutterEnv(cfg, mode="train")
    obs, info = env.reset(seed=42)
    
    steps = 0
    t0 = time.perf_counter()
    
    while steps < n_steps:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        
        if terminated or truncated:
            obs, info = env.reset()
            
    t1 = time.perf_counter()
    elapsed = t1 - t0
    steps_per_sec = steps / elapsed
    
    # 100 steps/s als weiches Kriterium (abhaengig von Hardware)
    passed = steps_per_sec >= 100.0  
    
    return {
        "passed": bool(passed),
        "steps_per_sec": float(steps_per_sec),
        "elapsed": float(elapsed),
        "total_steps": steps
    }


def run_all_checks():
    cfg = default_config()
    print(f"Starte Sanity Checks v{cfg.version}...")
    
    all_passed = True
    results = {}
    
    def run_test(name, func, *args):
        nonlocal all_passed
        print(f"Laufe {name} ... ", end="", flush=True)
        t0 = time.time()
        res = func(*args)
        t1 = time.time()
        
        if res.get("passed", False):
            print(f"PASS ({t1-t0:.2f}s)")
        else:
            print(f"FAIL ({t1-t0:.2f}s)")
            all_passed = False
            
        results[name] = res
    
    run_test("test_random_policy_smoke", test_random_policy_smoke, cfg)
    run_test("test_reward_decomposition", test_reward_decomposition, cfg)
    run_test("test_determinism", test_determinism, cfg)
    run_test("test_env_speed", test_env_speed, cfg)
    run_test("test_baseline_replay", test_baseline_replay, cfg)
    
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"sanity_v{cfg.version}.json"
    
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n--- Zusammenfassung ---")
    if all_passed:
        print("ALLE SANITY CHECKS BESTANDEN! Bereit fuer Phase 5.")
    else:
        print("FEHLER IN DEN SANITY CHECKS! Bitte `rl/results/sanity_*.json` ueberpruefen.")

if __name__ == "__main__":
    run_all_checks()
