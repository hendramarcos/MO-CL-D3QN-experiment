"""
single_intersection_experiment.py

Program lengkap pengujian Persimpangan Tunggal untuk menghasilkan 5 metrik evaluasi penelitian:
1) Avg Travel Time (s)
2) Avg Queue Length
3) Avg Delay / TimeLoss (s/veh)
4) Throughput (veh/h)
5) Fuel Consumption (L/h)

Metode:
- MO-CL-D3QN (Proposed)
- MO-D3QN (no curriculum)
- CL-DDQN (single objective)
- Single-Objective DDQN
- Actuated Controller
- Fixed-Time Controller

Contoh:
python single_intersection_experiment.py --mode full_pipeline --episodes 120 --eval-episodes 5
python single_intersection_experiment.py --mode full_pipeline --episodes 5 --eval-episodes 2 --max-steps 500
python single_intersection_experiment.py --mode baselines --eval-episodes 5
"""

import argparse
import csv
import os
import random
import sys
from collections import deque, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    if "SUMO_HOME" in os.environ:
        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)
    import traci
    from sumolib import checkBinary
except Exception as exc:
    raise RuntimeError("SUMO/TraCI tidak ditemukan. Pastikan SUMO_HOME sudah diset.") from exc


@dataclass
class Config:
    sumocfg: str = "scenario/environment.sumocfg"
    tls_id: Optional[str] = None
    output_dir: str = "outputs_single_intersection"
    max_steps: int = 3600
    decision_interval: int = 10
    yellow_duration: int = 3
    seed: int = 42
    episodes: int = 120
    eval_episodes: int = 5
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 64
    replay_size: int = 50000
    min_replay_size: int = 1000
    target_update_freq: int = 500
    hidden_dim: int = 256
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 25000
    stage1_end: int = 40
    stage2_end: int = 80
    observation_range_m: float = 100.0
    cell_length_m: float = 10.0
    max_lane_vehicles: float = 30.0
    max_lane_waiting: float = 300.0
    max_speed: float = 13.89
    wait_norm: float = 120.0
    queue_norm: float = 40.0
    throughput_norm: float = 20.0
    fuel_norm: float = 50000.0
    delay_norm: float = 120.0


METHODS = {
    "mo_cl_d3qn": {"label": "MO-CL-D3QN (Proposed)", "dueling": True, "double": True, "curriculum": True, "reward_type": "multi"},
    "mo_d3qn_no_curriculum": {"label": "MO-D3QN (no curriculum)", "dueling": True, "double": True, "curriculum": False, "reward_type": "multi"},
    "cl_ddqn_single_objective": {"label": "CL-DDQN (single objective)", "dueling": False, "double": True, "curriculum": True, "reward_type": "single"},
    "single_objective_ddqn": {"label": "Single-Objective DDQN", "dueling": False, "double": True, "curriculum": False, "reward_type": "single"},
    "actuated": {"label": "Actuated Controller", "controller": "actuated"},
    "fixed_time": {"label": "Fixed-Time Controller", "controller": "fixed_time"},
}

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_csv(path: Path, row: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def mean0(vals):
    return float(np.mean(vals)) if vals else 0.0


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int, dueling: bool):
        super().__init__()
        self.dueling = dueling
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        if dueling:
            self.value = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1))
            self.adv = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, action_dim))
        else:
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, action_dim))

    def forward(self, x):
        h = self.feature(x)
        if self.dueling:
            v = self.value(h)
            a = self.adv(h)
            return v + a - a.mean(dim=1, keepdim=True)
        return self.head(h)


class DDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, cfg: Config, method: str, device):
        self.cfg = cfg
        self.method = method
        self.method_cfg = METHODS[method]
        self.device = device
        self.action_dim = action_dim
        self.double = self.method_cfg.get("double", True)
        dueling = self.method_cfg.get("dueling", True)
        self.policy_net = QNetwork(state_dim, action_dim, cfg.hidden_dim, dueling).to(device)
        self.target_net = QNetwork(state_dim, action_dim, cfg.hidden_dim, dueling).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.replay = ReplayBuffer(cfg.replay_size)
        self.train_steps = 0

    def epsilon(self):
        frac = min(1.0, self.train_steps / self.cfg.epsilon_decay_steps)
        return self.cfg.epsilon_start + frac * (self.cfg.epsilon_end - self.cfg.epsilon_start)

    def select_action(self, state, eval_mode=False):
        if (not eval_mode) and random.random() < self.epsilon():
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            return int(torch.argmax(self.policy_net(s), dim=1).item())

    def optimize(self):
        if len(self.replay) < max(self.cfg.min_replay_size, self.cfg.batch_size):
            return None
        b = self.replay.sample(self.cfg.batch_size)
        states = torch.as_tensor(np.array(b.state), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(b.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(b.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(np.array(b.next_state), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(b.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        q_pred = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            if self.double:
                na = torch.argmax(self.policy_net(next_states), dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, na)
            else:
                next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            q_target = rewards + self.cfg.gamma * (1.0 - dones) * next_q
        loss = nn.SmoothL1Loss()(q_pred, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % self.cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "method": self.method,
        }, path)

    def load(self, path: Path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_state_dict"])
        self.target_net.load_state_dict(ckpt.get("target_state_dict", ckpt["policy_state_dict"]))
        if "optimizer_state_dict" in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception:
                pass
        self.train_steps = ckpt.get("train_steps", 0)


class SingleIntersectionEnv:
    def __init__(self, cfg: Config, gui=False, controller="rl", method="mo_cl_d3qn"):
        self.cfg = cfg
        self.gui = gui
        self.controller = controller
        self.method = method
        self.sumo_binary = checkBinary("sumo-gui" if gui else "sumo")
        self.tls_id = cfg.tls_id
        self.incoming_lanes = []
        self.green_phases = []
        self.yellow_after_green = {}
        self.current_green_phase = None
        self.n_cells = int(cfg.observation_range_m / cfg.cell_length_m)
        self.state_dim = None
        self.action_dim = None
        self.metrics = None
        self.depart_time = {}
        self.last_timeloss = {}
        self.travel_times = []
        self.delays = []
        self.sim_time = 0.0

    def _sumo_cmd(self, seed):
        return [self.sumo_binary, "-c", self.cfg.sumocfg, "--start", "--quit-on-end", "--no-step-log", "true", "--waiting-time-memory", "10000", "--seed", str(seed)]

    def reset(self, seed):
        if traci.isLoaded():
            traci.close(False)
        traci.start(self._sumo_cmd(seed))
        self._init_network_info()
        self.current_green_phase = self.green_phases[0]
        traci.trafficlight.setPhase(self.tls_id, self.current_green_phase)
        self.metrics = {"sum_queue_length": 0.0, "sum_running_waiting": 0.0, "sum_speed": 0.0, "sum_fuel_mg": 0.0, "throughput": 0, "steps": 0, "cumulative_reward": 0.0, "loss_sum": 0.0, "loss_count": 0}
        self.depart_time = {}
        self.last_timeloss = {}
        self.travel_times = []
        self.delays = []
        self.sim_time = 0.0
        return self._get_state()

    def close(self):
        if traci.isLoaded():
            traci.close(False)

    def _init_network_info(self):
        if self.tls_id is None:
            ids = list(traci.trafficlight.getIDList())
            if not ids:
                raise RuntimeError("Tidak ada traffic light pada network.")
            self.tls_id = ids[0]
        self.incoming_lanes = sorted(set(traci.trafficlight.getControlledLanes(self.tls_id)))
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        phases = logic.phases
        self.green_phases = []
        for i, ph in enumerate(phases):
            if (("G" in ph.state) or ("g" in ph.state)) and ("y" not in ph.state):
                self.green_phases.append(i)
        if not self.green_phases:
            raise RuntimeError(f"Tidak menemukan green phase pada TLS {self.tls_id}")
        self.yellow_after_green = {}
        for gp in self.green_phases:
            yp = None
            for k in range(1, min(4, len(phases)) + 1):
                idx = (gp + k) % len(phases)
                if "y" in phases[idx].state:
                    yp = idx
                    break
                if idx in self.green_phases:
                    break
            self.yellow_after_green[gp] = yp
        self.state_dim = len(self.incoming_lanes) * self.n_cells + len(self.incoming_lanes) * 5 + len(self.green_phases)
        self.action_dim = len(self.green_phases)

    def _is_done(self):
        return traci.simulation.getTime() >= self.cfg.max_steps or traci.simulation.getMinExpectedNumber() <= 0

    def _simulation_step_collect(self):
        traci.simulationStep()
        t = traci.simulation.getTime()
        self.sim_time = t
        for vid in traci.simulation.getDepartedIDList():
            self.depart_time[vid] = t
            self.last_timeloss[vid] = 0.0
        running = list(traci.vehicle.getIDList())
        speeds, waits = [], []
        fuel_mg_step = 0.0
        for vid in running:
            try:
                speeds.append(traci.vehicle.getSpeed(vid))
                waits.append(traci.vehicle.getWaitingTime(vid))
                self.last_timeloss[vid] = traci.vehicle.getTimeLoss(vid)
                fuel_mg_step += traci.vehicle.getFuelConsumption(vid)  # mg/s at step length 1s
            except Exception:
                pass
        for vid in traci.simulation.getArrivedIDList():
            self.metrics["throughput"] += 1
            if vid in self.depart_time:
                self.travel_times.append(t - self.depart_time.pop(vid))
            if vid in self.last_timeloss:
                self.delays.append(self.last_timeloss.pop(vid))
        queue = sum(traci.lane.getLastStepHaltingNumber(l) for l in self.incoming_lanes)
        self.metrics["sum_queue_length"] += float(queue)
        self.metrics["sum_running_waiting"] += mean0(waits)
        self.metrics["sum_speed"] += mean0(speeds)
        self.metrics["sum_fuel_mg"] += fuel_mg_step
        self.metrics["steps"] += 1

    def _phase_one_hot(self):
        x = np.zeros(len(self.green_phases), dtype=np.float32)
        if self.current_green_phase in self.green_phases:
            x[self.green_phases.index(self.current_green_phase)] = 1.0
        return x

    def _dtse(self):
        occ = np.zeros((len(self.incoming_lanes), self.n_cells), dtype=np.float32)
        for li, lane_id in enumerate(self.incoming_lanes):
            try:
                length = traci.lane.getLength(lane_id)
                vehs = traci.lane.getLastStepVehicleIDs(lane_id)
            except Exception:
                continue
            for vid in vehs:
                try:
                    pos = traci.vehicle.getLanePosition(vid)
                    dist = max(0.0, length - pos)
                    if dist <= self.cfg.observation_range_m:
                        cell = int(dist // self.cfg.cell_length_m)
                        if 0 <= cell < self.n_cells:
                            occ[li, cell] = 1.0
                except Exception:
                    pass
        return occ.flatten()

    def _lane_features(self):
        feats = []
        for lane_id in self.incoming_lanes:
            try:
                veh = traci.lane.getLastStepVehicleNumber(lane_id)
                halt = traci.lane.getLastStepHaltingNumber(lane_id)
                speed = traci.lane.getLastStepMeanSpeed(lane_id)
                wait = traci.lane.getWaitingTime(lane_id)
                occ = traci.lane.getLastStepOccupancy(lane_id)
            except Exception:
                veh, halt, speed, wait, occ = 0, 0, 0, 0, 0
            feats.extend([min(veh / self.cfg.max_lane_vehicles, 1.0), min(halt / self.cfg.max_lane_vehicles, 1.0), min(max(speed, 0.0) / self.cfg.max_speed, 1.0), min(wait / self.cfg.max_lane_waiting, 1.0), min(occ / 100.0, 1.0)])
        return np.asarray(feats, dtype=np.float32)

    def _get_state(self):
        return np.concatenate([self._dtse(), self._lane_features(), self._phase_one_hot()]).astype(np.float32)

    def _set_action_phase(self, action):
        next_green = self.green_phases[int(action)]
        if next_green == self.current_green_phase:
            traci.trafficlight.setPhase(self.tls_id, self.current_green_phase)
            return
        yellow = self.yellow_after_green.get(self.current_green_phase)
        if yellow is not None:
            traci.trafficlight.setPhase(self.tls_id, yellow)
            for _ in range(self.cfg.yellow_duration):
                if self._is_done():
                    break
                self._simulation_step_collect()
        self.current_green_phase = next_green
        traci.trafficlight.setPhase(self.tls_id, self.current_green_phase)

    def _actuated_action(self):
        links = traci.trafficlight.getControlledLinks(self.tls_id)
        logic = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        best_action, best_score = 0, -1.0
        for ai, phase_idx in enumerate(self.green_phases):
            state = logic.phases[phase_idx].state
            served = set()
            for sig_idx, sig in enumerate(state):
                if sig in ["G", "g"] and sig_idx < len(links):
                    for link in links[sig_idx]:
                        if len(link) > 0:
                            served.add(link[0])
            score = 0.0
            for lane in served:
                try:
                    score += traci.lane.getLastStepHaltingNumber(lane)
                    score += 0.05 * traci.lane.getWaitingTime(lane)
                except Exception:
                    pass
            if score > best_score:
                best_score, best_action = score, ai
        return best_action

    def _reward_parts(self, wait, queue, throughput, fuel_mg, delay):
        return {"r_wait": -min(wait / self.cfg.wait_norm, 2.0), "r_queue": -min(queue / self.cfg.queue_norm, 2.0), "r_throughput": min(throughput / self.cfg.throughput_norm, 2.0), "r_fuel": -min(fuel_mg / self.cfg.fuel_norm, 2.0), "r_delay": -min(delay / self.cfg.delay_norm, 2.0)}

    def _reward(self, parts, stage):
        m = METHODS.get(self.method, METHODS["mo_cl_d3qn"])
        if not m.get("curriculum", True):
            stage = 3
        if m.get("reward_type") == "single":
            return 0.70 * parts["r_delay"] + 0.30 * parts["r_queue"]
        if stage == 1:
            return 0.55 * parts["r_delay"] + 0.25 * parts["r_wait"] + 0.20 * parts["r_queue"]
        if stage == 2:
            return 0.35 * parts["r_delay"] + 0.20 * parts["r_wait"] + 0.20 * parts["r_queue"] + 0.25 * parts["r_throughput"]
        return 0.30 * parts["r_delay"] + 0.15 * parts["r_wait"] + 0.20 * parts["r_queue"] + 0.20 * parts["r_throughput"] + 0.15 * parts["r_fuel"]

    def step(self, action, stage):
        before = dict(self.metrics)
        before_delay_n = len(self.delays)
        if self.controller == "fixed_time":
            for _ in range(self.cfg.decision_interval):
                if self._is_done():
                    break
                self._simulation_step_collect()
        else:
            if self.controller == "actuated":
                action = self._actuated_action()
            self._set_action_phase(action)
            for _ in range(self.cfg.decision_interval):
                if self._is_done():
                    break
                self._simulation_step_collect()
        after = self.metrics
        interval_steps = max(1, after["steps"] - before["steps"])
        q = (after["sum_queue_length"] - before["sum_queue_length"]) / interval_steps
        w = (after["sum_running_waiting"] - before["sum_running_waiting"]) / interval_steps
        f = after["sum_fuel_mg"] - before["sum_fuel_mg"]
        tp = after["throughput"] - before["throughput"]
        new_delays = self.delays[before_delay_n:]
        d = mean0(new_delays) if new_delays else w
        parts = self._reward_parts(w, q, tp, f, d)
        reward = self._reward(parts, stage)
        self.metrics["cumulative_reward"] += reward
        done = self._is_done()
        ns = self._get_state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        return ns, float(reward), done, {"interval_wait": w, "interval_queue": q, "interval_throughput": tp, "interval_fuel_mg": f, "interval_delay": d, **parts}

    def summary(self):
        steps = max(1, self.metrics["steps"])
        sim_hours = max(self.sim_time, 1.0) / 3600.0
        total_fuel_l = self.metrics["sum_fuel_mg"] / 748900.0
        return {"avg_travel_time_s": mean0(self.travel_times), "avg_queue_length": self.metrics["sum_queue_length"] / steps, "avg_delay_s_per_veh": mean0(self.delays), "throughput_veh_h": self.metrics["throughput"] / sim_hours, "fuel_consumption_L_h": total_fuel_l / sim_hours, "avg_running_waiting_time_s": self.metrics["sum_running_waiting"] / steps, "avg_speed_m_s": self.metrics["sum_speed"] / steps, "arrived_vehicles": self.metrics["throughput"], "total_fuel_L": total_fuel_l, "sim_time_s": self.sim_time, "cumulative_reward": self.metrics["cumulative_reward"], "avg_loss": self.metrics["loss_sum"] / max(1, self.metrics["loss_count"])}


def stage_of_episode(ep, cfg):
    if ep <= cfg.stage1_end:
        return 1
    if ep <= cfg.stage2_end:
        return 2
    return 3


def train_method(cfg, method, gui=False):
    set_seed(cfg.seed)
    out = Path(cfg.output_dir) / method
    log_path = out / "training_metrics.csv"
    env = SingleIntersectionEnv(cfg, gui=gui, controller="rl", method=method)
    env.reset(seed=cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDQNAgent(env.state_dim, env.action_dim, cfg, method, device)
    env.close()
    best_score = -float("inf")
    print("=" * 80)
    print(f"Training {METHODS[method]['label']}")
    print(f"Output: {out}")
    print("=" * 80)
    try:
        for ep in range(1, cfg.episodes + 1):
            state = env.reset(seed=cfg.seed + ep)
            done = False
            stage = stage_of_episode(ep, cfg)
            while not done:
                action = agent.select_action(state, eval_mode=False)
                ns, reward, done, _ = env.step(action, stage)
                agent.replay.push(state, action, reward, ns, done)
                state = ns
                loss = agent.optimize()
                if loss is not None:
                    env.metrics["loss_sum"] += loss
                    env.metrics["loss_count"] += 1
            s = env.summary()
            row = {"method": method, "model": METHODS[method]["label"], "episode": ep, "stage": stage if METHODS[method].get("curriculum", True) else 0, "epsilon": agent.epsilon(), **s}
            append_csv(log_path, row)
            if s["cumulative_reward"] > best_score:
                best_score = s["cumulative_reward"]
                agent.save(out / "best_model.pt")
            agent.save(out / "last_model.pt")
            print(f"{METHODS[method]['label']} | EP {ep:03d} | ATT {s['avg_travel_time_s']:.2f}s | AQL {s['avg_queue_length']:.2f} | Delay {s['avg_delay_s_per_veh']:.2f}s/veh | TP {s['throughput_veh_h']:.2f} | Fuel {s['fuel_consumption_L_h']:.2f} | Loss {s['avg_loss']:.5f} | eps {agent.epsilon():.3f}")
    finally:
        env.close()
    print(f"Model terbaik: {out / 'best_model.pt'}")


def evaluate_rl_method(cfg, method, gui=False):
    out = Path(cfg.output_dir) / method
    model_path = out / "best_model.pt"
    env = SingleIntersectionEnv(cfg, gui=gui, controller="rl", method=method)
    env.reset(seed=cfg.seed + 999)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDQNAgent(env.state_dim, env.action_dim, cfg, method, device)
    agent.load(model_path)
    env.close()
    rows = []
    try:
        for ep in range(1, cfg.eval_episodes + 1):
            state = env.reset(seed=cfg.seed + 20000 + ep)
            done = False
            while not done:
                action = agent.select_action(state, eval_mode=True)
                state, _, done, _ = env.step(action, stage=3)
            s = env.summary()
            rows.append({"method": method, "model": METHODS[method]["label"], "eval_episode": ep, **s})
            print(f"EVAL {METHODS[method]['label']} | {ep:02d} | ATT {s['avg_travel_time_s']:.2f}s | AQL {s['avg_queue_length']:.2f} | Delay {s['avg_delay_s_per_veh']:.2f}s/veh | TP {s['throughput_veh_h']:.2f} | Fuel {s['fuel_consumption_L_h']:.2f}")
    finally:
        env.close()
    return rows


def evaluate_baseline(cfg, method, gui=False):
    env = SingleIntersectionEnv(cfg, gui=gui, controller=METHODS[method]["controller"], method="mo_cl_d3qn")
    rows = []
    try:
        for ep in range(1, cfg.eval_episodes + 1):
            env.reset(seed=cfg.seed + 30000 + ep)
            done = False
            while not done:
                _, _, done, _ = env.step(None, stage=3)
            s = env.summary()
            rows.append({"method": method, "model": METHODS[method]["label"], "eval_episode": ep, **s})
            print(f"EVAL {METHODS[method]['label']} | {ep:02d} | ATT {s['avg_travel_time_s']:.2f}s | AQL {s['avg_queue_length']:.2f} | Delay {s['avg_delay_s_per_veh']:.2f}s/veh | TP {s['throughput_veh_h']:.2f} | Fuel {s['fuel_consumption_L_h']:.2f}")
    finally:
        env.close()
    return rows


def summarize_results(rows, cfg):
    df = pd.DataFrame(rows)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "evaluation_raw.csv", index=False)
    summary = df.groupby(["method", "model"], as_index=False).agg(avg_travel_time_s=("avg_travel_time_s", "mean"), avg_queue_length=("avg_queue_length", "mean"), avg_delay_s_per_veh=("avg_delay_s_per_veh", "mean"), throughput_veh_h=("throughput_veh_h", "mean"), fuel_consumption_L_h=("fuel_consumption_L_h", "mean"))
    order = ["mo_cl_d3qn", "mo_d3qn_no_curriculum", "cl_ddqn_single_objective", "single_objective_ddqn", "actuated", "fixed_time"]
    summary["order"] = summary["method"].apply(lambda x: order.index(x) if x in order else 999)
    summary = summary.sort_values("order").drop(columns=["order"])
    table = summary[["model", "avg_travel_time_s", "avg_queue_length", "avg_delay_s_per_veh", "throughput_veh_h", "fuel_consumption_L_h"]].copy()
    table.columns = ["Model", "Avg Travel Time (s)", "Avg Queue Length", "Avg Delay (s/veh)", "Throughput (veh/h)", "Fuel Consumption (L/h)"]
    for c in table.columns[1:]:
        table[c] = table[c].round(2)
    table.to_csv(out / "evaluation_summary.csv", index=False)
    summary.to_csv(out / "evaluation_summary_unrounded.csv", index=False)
    print("\n" + "=" * 80)
    print("lima metrik evaluasi penelitian Perbandingan Kinerja Metode")
    print("=" * 80)
    print(table.to_string(index=False))
    print("=" * 80)
    print(f"CSV: {out / 'evaluation_summary.csv'}")
    return table


def run_training_all(cfg, gui=False):
    for m in ["mo_cl_d3qn", "mo_d3qn_no_curriculum", "cl_ddqn_single_objective", "single_objective_ddqn"]:
        train_method(cfg, m, gui=gui)


def run_evaluation_all(cfg, gui=False):
    rows = []
    for m in ["mo_cl_d3qn", "mo_d3qn_no_curriculum", "cl_ddqn_single_objective", "single_objective_ddqn"]:
        if (Path(cfg.output_dir) / m / "best_model.pt").exists():
            rows.extend(evaluate_rl_method(cfg, m, gui=gui))
        else:
            print(f"SKIP {m}: model belum ditemukan.")
    for m in ["actuated", "fixed_time"]:
        rows.extend(evaluate_baseline(cfg, m, gui=gui))
    return summarize_results(rows, cfg)


def run_baselines(cfg, gui=False):
    rows = []
    for m in ["actuated", "fixed_time"]:
        rows.extend(evaluate_baseline(cfg, m, gui=gui))
    return summarize_results(rows, cfg)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="full_pipeline", choices=["full_pipeline", "train_all", "evaluate_all", "baselines", "train", "evaluate"])
    p.add_argument("--method", default="mo_cl_d3qn", choices=list(METHODS.keys()))
    p.add_argument("--sumocfg", default="scenario/environment.sumocfg")
    p.add_argument("--tls-id", default=None)
    p.add_argument("--output-dir", default="outputs_single_intersection")
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--decision-interval", type=int, default=10)
    p.add_argument("--yellow-duration", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gui", action="store_true")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--stage1-end", type=int, default=40)
    p.add_argument("--stage2-end", type=int, default=80)
    return p.parse_args()


def main():
    a = parse_args()
    cfg = Config(sumocfg=a.sumocfg, tls_id=a.tls_id, output_dir=a.output_dir, episodes=a.episodes, eval_episodes=a.eval_episodes, max_steps=a.max_steps, decision_interval=a.decision_interval, yellow_duration=a.yellow_duration, seed=a.seed, learning_rate=a.lr, batch_size=a.batch_size, stage1_end=a.stage1_end, stage2_end=a.stage2_end)
    if a.mode == "full_pipeline":
        run_training_all(cfg, gui=a.gui)
        run_evaluation_all(cfg, gui=a.gui)
    elif a.mode == "train_all":
        run_training_all(cfg, gui=a.gui)
    elif a.mode == "evaluate_all":
        run_evaluation_all(cfg, gui=a.gui)
    elif a.mode == "baselines":
        run_baselines(cfg, gui=a.gui)
    elif a.mode == "train":
        if a.method in ["actuated", "fixed_time"]:
            raise ValueError("Baseline tidak perlu training. Gunakan --mode baselines atau --mode evaluate.")
        train_method(cfg, a.method, gui=a.gui)
    elif a.mode == "evaluate":
        rows = evaluate_baseline(cfg, a.method, gui=a.gui) if a.method in ["actuated", "fixed_time"] else evaluate_rl_method(cfg, a.method, gui=a.gui)
        summarize_results(rows, cfg)


if __name__ == "__main__":
    main()
