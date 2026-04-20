import argparse
import json
import os

# Work around numba cache issues in some headless / conda environments.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import h5py
import imageio
import numpy as np

import init_path


def _to_uint8(frame):
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 1.0) * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def _sorted_demo_keys(group):
    keys = list(group.keys())

    def key_fn(k):
        try:
            return int(k.split("_")[-1])
        except Exception:
            return k

    return sorted(keys, key=key_fn)


def export_processed_demo(
    f, demo_key, output_path, camera_key="agentview_rgb", fps=20, max_frames=-1
):
    demo = f["data"][demo_key]
    if "obs" not in demo or camera_key not in demo["obs"]:
        raise ValueError(
            f"Processed demo requires data/{demo_key}/obs/{camera_key}, but it was not found"
        )

    frames = demo["obs"][camera_key][()]
    if max_frames > 0:
        frames = frames[:max_frames]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(_to_uint8(frame))


def export_raw_demo(
    f,
    demo_key,
    output_path,
    fps=20,
    max_frames=-1,
    raw_camera_name="agentview",
    raw_obs_key="agentview_image",
    height=128,
    width=128,
):
    import libero.libero.utils.utils as libero_utils
    from libero.libero.envs import TASK_MAPPING

    problem_info = json.loads(f["data"].attrs["problem_info"])
    env_kwargs = json.loads(f["data"].attrs["env_info"])
    bddl_file_name = f["data"].attrs["bddl_file_name"]

    env_kwargs = dict(env_kwargs)
    env_kwargs.update(
        {
            "bddl_file_name": bddl_file_name,
            "has_renderer": False,
            "has_offscreen_renderer": True,
            "ignore_done": True,
            "use_camera_obs": True,
            "camera_depths": False,
            "camera_names": [raw_camera_name],
            "camera_heights": height,
            "camera_widths": width,
            "reward_shaping": True,
            "control_freq": 20,
        }
    )

    problem_name = problem_info["problem_name"]
    env = TASK_MAPPING[problem_name](**env_kwargs)

    demo = f["data"][demo_key]
    states = demo["states"][()]
    actions = demo["actions"][()]
    model_xml = libero_utils.postprocess_model_xml(demo.attrs["model_file"], {})

    reset_success = False
    while not reset_success:
        try:
            env.reset()
            reset_success = True
        except Exception:
            continue

    env.reset_from_xml_string(model_xml)
    env.sim.reset()
    env.sim.set_state_from_flattened(states[0])
    env.sim.forward()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        steps = len(actions)
        if max_frames > 0:
            steps = min(steps, max_frames)

        for i in range(steps):
            obs, _, _, _ = env.step(actions[i])
            frame = obs[raw_obs_key]
            writer.append_data(_to_uint8(frame))

    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Export a demo trajectory from HDF5 to MP4 video."
    )
    parser.add_argument("--dataset", required=True, help="Path to .hdf5 file")
    parser.add_argument(
        "--demo-key",
        default=None,
        help="Demo key like demo_1. Default: the first demo in data/",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output .mp4 path. Default: <dataset_basename>_<demo_key>.mp4",
    )
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--max-frames", type=int, default=-1, help="Limit number of frames"
    )

    parser.add_argument(
        "--camera-key",
        default="agentview_rgb",
        help="For processed datasets: obs key, e.g. agentview_rgb",
    )
    parser.add_argument(
        "--raw-camera-name",
        default="agentview",
        help="For raw datasets: camera name passed to env",
    )
    parser.add_argument(
        "--raw-obs-key",
        default="agentview_image",
        help="For raw datasets: observation key, e.g. agentview_image",
    )
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)

    args = parser.parse_args()

    with h5py.File(args.dataset, "r") as f:
        if "data" not in f:
            raise ValueError("Invalid dataset: missing 'data' group")

        demo_keys = _sorted_demo_keys(f["data"])
        if len(demo_keys) == 0:
            raise ValueError("Invalid dataset: no demo_* groups under data/")

        demo_key = args.demo_key or demo_keys[0]
        if demo_key not in f["data"]:
            raise ValueError(f"Demo key {demo_key} not found. Available: {demo_keys[:5]}...")

        output = args.output
        if output is None:
            dataset_name = os.path.splitext(os.path.basename(args.dataset))[0]
            output = f"{dataset_name}_{demo_key}.mp4"

        demo = f["data"][demo_key]
        is_processed = "obs" in demo

        if is_processed:
            export_processed_demo(
                f,
                demo_key,
                output,
                camera_key=args.camera_key,
                fps=args.fps,
                max_frames=args.max_frames,
            )
            print(f"[info] exported processed demo video: {output}")
        else:
            export_raw_demo(
                f,
                demo_key,
                output,
                fps=args.fps,
                max_frames=args.max_frames,
                raw_camera_name=args.raw_camera_name,
                raw_obs_key=args.raw_obs_key,
                height=args.height,
                width=args.width,
            )
            print(f"[info] exported raw demo playback video: {output}")


if __name__ == "__main__":
    main()
