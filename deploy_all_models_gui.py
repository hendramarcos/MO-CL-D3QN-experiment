"""
deploy_all_models_gui.py

Kode tambahan untuk IMPLEMENTASI semua model lima metrik evaluasi penelitian ke SUMO GUI
pada skenario Persimpangan Tunggal.

Model/metode yang didukung:
1. MO-CL-D3QN (Proposed)
2. MO-D3QN (no curriculum)
3. CL-DDQN (single objective)
4. Single-Objective DDQN
5. Actuated Controller
6. Fixed-Time Controller

Syarat:
- Letakkan file ini dalam folder yang sama dengan:
  single_intersection_experiment.py
  environment.sumocfg
  environment.net.xml
  routes_medium.rou.xml
- Model hasil training sudah tersedia pada folder:
  outputs_single_intersection/<nama_method>/best_model.pt

Contoh menjalankan semua model ke SUMO GUI:
python deploy_all_models_gui.py --method all --gui --wait-before-start --keep-open

Contoh menjalankan model proposed saja:
python deploy_all_models_gui.py --method mo_cl_d3qn --gui --wait-before-start --keep-open

Contoh menjalankan baseline fixed-time saja:
python deploy_all_models_gui.py --method fixed_time --gui --wait-before-start --keep-open

Output:
outputs_deployment/
  deployment_all_models_metrics.csv
  deployment_actions_<method>.csv
"""

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

# Import dari program eksperimen utama.
from single_intersection_experiment import (
    Config,
    METHODS,
    SingleIntersectionEnv,
    DDQNAgent,
    set_seed,
)


RL_METHODS = [
    "mo_cl_d3qn",
    "mo_d3qn_no_curriculum",
    "cl_ddqn_single_objective",
    "single_objective_ddqn",
]

BASELINE_METHODS = [
    "actuated",
    "fixed_time",
]

ALL_METHODS = RL_METHODS + BASELINE_METHODS


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def resolve_model_path(method: str, model_root: str, model_name: str) -> Path:
    return Path(model_root) / method / model_name


def make_env(cfg: Config, method: str, gui: bool):
    if method == "actuated":
        return SingleIntersectionEnv(cfg, gui=gui, controller="actuated", method="mo_cl_d3qn")
    if method == "fixed_time":
        return SingleIntersectionEnv(cfg, gui=gui, controller="fixed_time", method="mo_cl_d3qn")
    return SingleIntersectionEnv(cfg, gui=gui, controller="rl", method=method)


def load_trained_agent(env, cfg: Config, method: str, model_path: Path, device):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: {model_path}\n"
            f"Jalankan training terlebih dahulu, contoh:\n"
            f"python single_intersection_experiment.py --mode train --method {method} --episodes 120"
        )

    agent = DDQNAgent(env.state_dim, env.action_dim, cfg, method, device)
    agent.load(model_path)
    agent.policy_net.eval()
    agent.target_net.eval()
    return agent


def current_phase_info(env):
    try:
        import traci
        phase_idx = traci.trafficlight.getPhase(env.tls_id)
        phase_name = ""
        try:
            logic = traci.trafficlight.getAllProgramLogics(env.tls_id)[0]
            if 0 <= phase_idx < len(logic.phases):
                phase_name = getattr(logic.phases[phase_idx], "name", "")
                phase_state = logic.phases[phase_idx].state
            else:
                phase_state = ""
        except Exception:
            phase_state = ""
        return phase_idx, phase_state, phase_name
    except Exception:
        return -1, "", ""


def deploy_one_method(
    cfg: Config,
    method: str,
    gui: bool,
    model_root: str,
    model_name: str,
    output_dir: Path,
    wait_before_start: bool,
    keep_open: bool,
    print_every: int,
    sleep_step: float,
):
    method_cfg = METHODS[method]
    label = method_cfg["label"]
    is_rl = method in RL_METHODS

    env = make_env(cfg, method, gui=gui)

    # reset membuka SUMO dan menyiapkan state awal.
    state = env.reset(seed=cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = None
    model_path = ""

    if is_rl:
        model_path = resolve_model_path(method, model_root, model_name)
        agent = load_trained_agent(env, cfg, method, model_path, device)

    print("\n" + "=" * 90)
    print(f"IMPLEMENTASI KE SUMO GUI: {label}")
    print("=" * 90)
    print(f"Method     : {method}")
    print(f"SUMO config: {cfg.sumocfg}")
    print(f"TLS ID     : {env.tls_id}")
    print(f"State dim  : {env.state_dim}")
    print(f"Action dim : {env.action_dim}")
    print(f"GUI        : {gui}")
    if is_rl:
        print(f"Model      : {model_path}")
    else:
        print("Model      : baseline controller, tidak memakai file .pt")
    print("=" * 90)

    if wait_before_start:
        input("SUMO GUI sudah terbuka pada time 0. Tekan ENTER untuk mulai implementasi model...")

    actions_log = output_dir / f"deployment_actions_{method}.csv"
    decision = 0
    done = False

    try:
        while not done:
            if method == "fixed_time":
                action = None
                action_label = "fixed_program"
            elif method == "actuated":
                action = None
                action_label = "actuated_heuristic"
            else:
                action = agent.select_action(state, eval_mode=True)
                action_label = f"rl_action_{action}"

            next_state, reward, done, info = env.step(action, stage=3)
            phase_idx, phase_state, phase_name = current_phase_info(env)

            action_row = {
                "method": method,
                "model": label,
                "decision": decision,
                "sim_time_s": env.sim_time,
                "action": action if action is not None else "",
                "action_label": action_label,
                "tls_id": env.tls_id,
                "phase_index": phase_idx,
                "phase_state": phase_state,
                "phase_name": phase_name,
                "reward": reward,
                "interval_wait": info.get("interval_wait", ""),
                "interval_queue": info.get("interval_queue", ""),
                "interval_throughput": info.get("interval_throughput", ""),
                "interval_fuel_mg": info.get("interval_fuel_mg", ""),
                "interval_delay": info.get("interval_delay", ""),
            }
            append_csv(actions_log, action_row)

            state = next_state
            decision += 1

            if decision % print_every == 0:
                summ = env.summary()
                print(
                    f"{label} | Decision {decision:04d} | "
                    f"t={env.sim_time:.0f}s | "
                    f"ATT {summ['avg_travel_time_s']:.2f}s | "
                    f"AQL {summ['avg_queue_length']:.2f} | "
                    f"Delay {summ['avg_delay_s_per_veh']:.2f}s/veh | "
                    f"TP {summ['throughput_veh_h']:.2f} veh/h | "
                    f"Fuel {summ['fuel_consumption_L_h']:.2f} L/h | "
                    f"Phase {phase_idx}"
                )

            if sleep_step > 0:
                time.sleep(sleep_step)

        summary = env.summary()
        result_row = {
            "method": method,
            "model": label,
            "model_path": str(model_path),
            "sumocfg": cfg.sumocfg,
            "seed": cfg.seed,
            "decision_steps": decision,
            **summary,
        }

        append_csv(output_dir / "deployment_all_models_metrics.csv", result_row)

        print("\n" + "-" * 90)
        print(f"IMPLEMENTASI SELESAI: {label}")
        print("-" * 90)
        print(f"Avg Travel Time      : {summary['avg_travel_time_s']:.2f} s")
        print(f"Avg Queue Length     : {summary['avg_queue_length']:.2f}")
        print(f"Avg Delay            : {summary['avg_delay_s_per_veh']:.2f} s/veh")
        print(f"Throughput           : {summary['throughput_veh_h']:.2f} veh/h")
        print(f"Fuel Consumption     : {summary['fuel_consumption_L_h']:.2f} L/h")
        print(f"Action log           : {actions_log}")
        print("-" * 90)

        if keep_open:
            input("Tekan ENTER untuk menutup SUMO GUI dan lanjut/keluar...")

        return result_row

    finally:
        env.close()


def deploy_all(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config(
        sumocfg=args.sumocfg,
        tls_id=args.tls_id,
        output_dir=args.model_root,
        max_steps=args.max_steps,
        decision_interval=args.decision_interval,
        yellow_duration=args.yellow_duration,
        seed=args.seed,
        learning_rate=args.lr,
        batch_size=args.batch_size,
    )

    set_seed(args.seed)

    if args.method == "all":
        methods = ALL_METHODS
    elif args.method == "rl_all":
        methods = RL_METHODS
    elif args.method == "baseline_all":
        methods = BASELINE_METHODS
    else:
        methods = [args.method]

    results = []
    for i, method in enumerate(methods):
        cfg.seed = args.seed + i * args.seed_offset
        result = deploy_one_method(
            cfg=cfg,
            method=method,
            gui=args.gui,
            model_root=args.model_root,
            model_name=args.model_name,
            output_dir=output_dir,
            wait_before_start=args.wait_before_start,
            keep_open=args.keep_open,
            print_every=args.print_every,
            sleep_step=args.sleep_step,
        )
        results.append(result)

    # Buat tabel ringkas 5 metrik seperti lima metrik evaluasi penelitian.
    if results:
        import pandas as pd

        df = pd.DataFrame(results)
        table = df[[
            "model",
            "avg_travel_time_s",
            "avg_queue_length",
            "avg_delay_s_per_veh",
            "throughput_veh_h",
            "fuel_consumption_L_h",
        ]].copy()

        table.columns = [
            "Model",
            "Avg Travel Time (s)",
            "Avg Queue Length",
            "Avg Delay (s/veh)",
            "Throughput (veh/h)",
            "Fuel Consumption (L/h)",
        ]

        for col in table.columns[1:]:
            table[col] = table[col].round(2)

        table_path = output_dir / "deployment_comparison_summary.csv"
        table.to_csv(table_path, index=False)

        print("\n" + "=" * 90)
        print("RINGKASAN IMPLEMENTASI SEMUA MODEL KE SUMO GUI")
        print("=" * 90)
        print(table.to_string(index=False))
        print("=" * 90)
        print(f"Ringkasan metrik: {output_dir / 'deployment_all_models_metrics.csv'}")
        print(f"Tabel 5 metrik  : {table_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy semua model lima metrik evaluasi penelitian ke SUMO GUI")
    parser.add_argument(
        "--method",
        default="all",
        choices=["all", "rl_all", "baseline_all"] + ALL_METHODS,
        help="Pilih model/metode yang dijalankan.",
    )
    parser.add_argument("--sumocfg", default="scenario/environment.sumocfg")
    parser.add_argument("--tls-id", default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--model-root", default="outputs_single_intersection")
    parser.add_argument("--model-name", default="best_model.pt")
    parser.add_argument("--output-dir", default="outputs_deployment")
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--decision-interval", type=int, default=10)
    parser.add_argument("--yellow-duration", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-offset", type=int, default=100)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--sleep-step", type=float, default=0.0, help="Tambahan jeda real-time antar decision untuk observasi GUI.")
    parser.add_argument("--wait-before-start", action="store_true", help="Menunggu ENTER sebelum simulasi mulai melangkah.")
    parser.add_argument("--keep-open", action="store_true", help="Menahan GUI tetap terbuka setelah satu metode selesai.")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--check-import", action="store_true", help="Cek kompatibilitas import tanpa menjalankan SUMO.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.check_import:
        print("IMPORT OK: Config, METHODS, SingleIntersectionEnv, DDQNAgent, dan set_seed berhasil dimuat.")
        print("Silakan lanjutkan deployment ke SUMO GUI.")
    else:
        deploy_all(args)
