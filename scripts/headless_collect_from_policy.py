import argparse
import datetime
import json
import os
import sys
from pathlib import Path

# Work around numba cache issues in some headless / conda environments.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import h5py
import numpy as np
import torch

import init_path

BENCHMARK_MAP = {
    "libero_10": "LIBERO_10",
    "libero_spatial": "LIBERO_SPATIAL",
    "libero_object": "LIBERO_OBJECT",
    "libero_goal": "LIBERO_GOAL",
}

ALGO_DIR_MAP = {
    "base": "Sequential",
    "er": "ER",
    "ewc": "EWC",
    "packnet": "PackNet",
    "multitask": "Multitask",
}

POLICY_DIR_MAP = {
    "bc_rnn_policy": "BCRNNPolicy",
    "bc_transformer_policy": "BCTransformerPolicy",
    "bc_vilt_policy": "BCViLTPolicy",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect demonstration trajectories headlessly from a trained policy."
    )
    parser.add_argument("--experiment-dir", type=str, default="experiments")
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["libero_10", "libero_spatial", "libero_object", "libero_goal"],
    )
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["base", "er", "ewc", "packnet", "multitask"],
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        choices=["bc_rnn_policy", "bc_transformer_policy", "bc_vilt_policy"],
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ep", type=int)
    parser.add_argument("--load_task", type=int)

    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--num-demos", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--post-success-steps", type=int, default=10)
    parser.add_argument("--allow-failures", action="store_true")
    parser.add_argument("--random-init", action="store_true")

    parser.add_argument("--controller", type=str, default="OSC_POSE")
    parser.add_argument("--robots", nargs="+", type=str, default=["Panda"])
    parser.add_argument("--config", type=str, default="single-arm-opposed")
    parser.add_argument("--img-height", type=int, default=128)
    parser.add_argument("--img-width", type=int, default=128)

    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    if args.algo == "multitask" and args.ep is None:
        raise ValueError("--ep is required when --algo multitask")
    if args.algo != "multitask" and args.load_task is None:
        raise ValueError("--load_task is required for non-multitask algorithms")

    return args


def find_latest_run_folder(experiment_dir):
    experiment_id = 0
    for path in Path(experiment_dir).glob("run_*"):
        if not path.is_dir():
            continue
        try:
            folder_id = int(str(path).split("run_")[-1])
            if folder_id > experiment_id:
                experiment_id = folder_id
        except Exception:
            pass

    if experiment_id == 0:
        return None
    return os.path.join(experiment_dir, f"run_{experiment_id:03d}")


def main():
    args = parse_args()

    import robosuite as suite

    import libero.libero.envs.bddl_utils as BDDLUtils
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
    from libero.lifelong.algos import EWC, ER, Multitask, PackNet, Sequential
    from libero.lifelong.main import get_task_embs
    from libero.lifelong.metric import raw_obs_to_tensor_obs
    from libero.lifelong.utils import (
        NpEncoder,
        control_seed,
        safe_device,
        torch_load_model,
    )

    algo_class_map = {
        "base": Sequential,
        "er": ER,
        "ewc": EWC,
        "packnet": PackNet,
        "multitask": Multitask,
    }

    if args.device_id < 0:
        device = "cpu"
    else:
        device = f"cuda:{args.device_id}"

    exp_dir = os.path.join(
        args.experiment_dir,
        f"{BENCHMARK_MAP[args.benchmark]}/{ALGO_DIR_MAP[args.algo]}/"
        + f"{POLICY_DIR_MAP[args.policy]}_seed{args.seed}",
    )
    run_folder = find_latest_run_folder(exp_dir)
    if run_folder is None:
        print(f"[error] cannot find checkpoint under {exp_dir}")
        sys.exit(1)

    if args.algo == "multitask":
        model_path = os.path.join(run_folder, f"multitask_model_ep{args.ep}.pth")
    else:
        model_path = os.path.join(run_folder, f"task{args.load_task}_model.pth")

    try:
        sd, cfg, previous_mask = torch_load_model(model_path, map_location=device)
    except Exception as exc:
        print(f"[error] failed to load model at {model_path}: {exc}")
        sys.exit(1)

    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    cfg.device = device

    algo = safe_device(algo_class_map[args.algo](10, cfg), cfg.device)
    if previous_mask is not None:
        algo.policy.previous_mask = previous_mask

    if cfg.lifelong.algo == "PackNet":
        algo.eval()
        for module_idx, module in enumerate(algo.policy.modules()):
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                weight = module.weight.data
                mask = algo.previous_masks[module_idx].to(cfg.device)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(args.task_id + 1)] = 0.0
            if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()

    algo.policy.load_state_dict(sd)
    algo.eval()

    if not hasattr(cfg.data, "task_order_index"):
        cfg.data.task_order_index = 0

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    if not (0 <= args.task_id < benchmark.n_tasks):
        raise ValueError(f"task_id should be in [0, {benchmark.n_tasks - 1}]")

    descriptions = [benchmark.get_task(i).language for i in range(benchmark.n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark.set_task_embs(task_embs)

    task = benchmark.get_task(args.task_id)
    task_emb = benchmark.get_task_emb(args.task_id)

    bddl_file_name = os.path.join(cfg.bddl_folder, task.problem_folder, task.bddl_file)
    problem_info = BDDLUtils.get_problem_info(bddl_file_name)

    controller_config = suite.load_controller_config(default_controller=args.controller)
    env_cfg = {
        "robots": args.robots,
        "controller_configs": controller_config,
    }
    if "TwoArm" in problem_info["problem_name"]:
        env_cfg["env_configuration"] = args.config

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file_name,
        **env_cfg,
        camera_heights=args.img_height,
        camera_widths=args.img_width,
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        control_freq=20,
    )

    init_states_path = os.path.join(
        cfg.init_states_folder, task.problem_folder, task.init_states_file
    )
    init_states = torch.load(init_states_path)
    if isinstance(init_states, torch.Tensor):
        init_states = init_states.cpu().numpy()

    max_steps = args.max_steps if args.max_steps is not None else cfg.eval.max_steps

    control_seed(args.seed)

    episodes = []
    attempts = 0
    init_idx = 0

    while len(episodes) < args.num_demos and attempts < args.max_attempts:
        attempts += 1

        env.reset()
        if args.random_init:
            idx = np.random.randint(init_states.shape[0])
        else:
            idx = init_idx % init_states.shape[0]
            init_idx += 1

        obs = env.set_init_state(init_states[idx])
        model_xml = env.sim.model.get_xml()

        algo.reset()

        actions = []
        states = []
        success = False

        for _ in range(max_steps):
            data = raw_obs_to_tensor_obs([obs], task_emb, cfg)
            action = algo.policy.get_action(data)[0]
            obs, _, _, _ = env.step(action)

            actions.append(np.asarray(action, dtype=np.float32))
            states.append(np.asarray(env.get_sim_state(), dtype=np.float64))

            if env.check_success():
                success = True
                break

        if success and args.post_success_steps > 0:
            for _ in range(args.post_success_steps):
                if len(actions) >= max_steps:
                    break
                data = raw_obs_to_tensor_obs([obs], task_emb, cfg)
                action = algo.policy.get_action(data)[0]
                obs, _, _, _ = env.step(action)
                actions.append(np.asarray(action, dtype=np.float32))
                states.append(np.asarray(env.get_sim_state(), dtype=np.float64))

        if success or args.allow_failures:
            episodes.append(
                {
                    "actions": np.asarray(actions, dtype=np.float32),
                    "states": np.asarray(states, dtype=np.float64),
                    "model_file": model_xml,
                    "success": success,
                }
            )
            print(
                f"[info] collected {len(episodes)}/{args.num_demos} | "
                f"attempt={attempts} | success={success} | steps={len(actions)}"
            )
        else:
            print(f"[info] skip failed attempt {attempts} | steps={len(actions)}")

    env.close()

    if len(episodes) == 0:
        print("[error] no demonstrations collected")
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with h5py.File(args.output, "w") as f:
        grp = f.create_group("data")
        now = datetime.datetime.now()
        grp.attrs["date"] = f"{now.month}-{now.day}-{now.year}"
        grp.attrs["time"] = f"{now.hour}:{now.minute}:{now.second}"
        grp.attrs["repository_version"] = suite.__version__
        grp.attrs["env"] = problem_info["problem_name"]
        grp.attrs["env_info"] = json.dumps(env_cfg, cls=NpEncoder)
        grp.attrs["problem_info"] = json.dumps(problem_info)
        grp.attrs["bddl_file_name"] = bddl_file_name
        with open(bddl_file_name, "r", encoding="utf-8") as fh:
            grp.attrs["bddl_file_content"] = fh.read()

        for i, ep in enumerate(episodes, 1):
            ep_grp = grp.create_group(f"demo_{i}")
            ep_grp.attrs["model_file"] = ep["model_file"]
            ep_grp.attrs["success"] = int(ep["success"])
            ep_grp.create_dataset("states", data=ep["states"])
            ep_grp.create_dataset("actions", data=ep["actions"])

    print(f"[info] saved {len(episodes)} demos to {args.output}")


if __name__ == "__main__":
    main()
