import argparse
import json
import os
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
from tqdm import tqdm


def _to_serializable(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _write_attrs(attrs, output_dir):
    payload = {key: _to_serializable(attrs[key]) for key in attrs.keys()}
    _write_json(os.path.join(output_dir, "attrs.json"), payload)


def _normalize_frame(frame):
    frame = np.asarray(frame)
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame, 0.0, 255.0)
        if frame.max() <= 1.0:
            frame = frame * 255.0
    return frame.astype(np.uint8)


def _looks_like_image_batch(array):
    if array.ndim != 4:
        return False
    if array.shape[-1] not in (1, 3, 4):
        return False
    return np.issubdtype(array.dtype, np.integer) or np.issubdtype(
        array.dtype, np.floating
    )


def _save_image_batch(array, output_dir, fps, save_frames):
    os.makedirs(output_dir, exist_ok=True)
    video_path = os.path.join(output_dir, "video.mp4")
    with imageio.get_writer(video_path, fps=fps) as writer:
        for idx, frame in enumerate(
            tqdm(array, desc=f"frames:{Path(output_dir).name}", leave=False)
        ):
            frame = _normalize_frame(frame)
            if save_frames:
                imageio.imwrite(os.path.join(output_dir, f"{idx:05d}.png"), frame)
            writer.append_data(frame)


def _unpack_dataset(name, dataset, output_dir, export_images, fps, save_frames):
    array = dataset[()]
    np.save(os.path.join(output_dir, f"{name}.npy"), array)

    meta = {
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
    }
    _write_json(os.path.join(output_dir, f"{name}.meta.json"), meta)

    if export_images and _looks_like_image_batch(array):
        _save_image_batch(
            array, os.path.join(output_dir, f"{name}_frames"), fps, save_frames
        )


def _unpack_group(group, output_dir, export_images, fps, save_frames):
    os.makedirs(output_dir, exist_ok=True)
    _write_attrs(group.attrs, output_dir)

    for key in tqdm(list(group.keys()), desc=f"group:{Path(output_dir).name}", leave=False):
        item = group[key]
        if isinstance(item, h5py.Group):
            _unpack_group(
                item, os.path.join(output_dir, key), export_images, fps, save_frames
            )
        elif isinstance(item, h5py.Dataset):
            _unpack_dataset(key, item, output_dir, export_images, fps, save_frames)


def _default_output_dir(dataset_path, output_root=None):
    dataset_path = Path(dataset_path)
    root = dataset_path.parent if output_root is None else Path(output_root)
    return root / dataset_path.stem


def _unpack_one_file(
    dataset_path, demo_key, output_root, export_images, fps, save_frames
):
    output_dir = _default_output_dir(dataset_path, output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        if demo_key is None:
            _unpack_group(f, str(output_dir), export_images, fps, save_frames)
        else:
            if "data" not in f or demo_key not in f["data"]:
                raise ValueError(f"Demo key {demo_key} not found under data/")
            _unpack_group(
                f["data"][demo_key],
                str(output_dir / demo_key),
                export_images,
                fps,
                save_frames,
            )
            _write_attrs(f["data"].attrs, str(output_dir))

    print(f"[info] unpacked {dataset_path} -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Unpack LIBERO HDF5 datasets into JSON, NPY, PNG, and MP4 files."
    )
    parser.add_argument("--dataset", help="Path to one .hdf5 file")
    parser.add_argument("--input-dir", help="Directory containing .hdf5 files")
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory for unpacked outputs. Default: next to each hdf5 file.",
    )
    parser.add_argument(
        "--demo-key",
        default=None,
        help="Optional demo key like demo_0 or demo_1. If omitted, unpack the whole file.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="FPS for exported videos from image arrays.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Do not export MP4 videos for image tensors.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Also export PNG frames in addition to MP4 videos.",
    )
    args = parser.parse_args()

    if bool(args.dataset) == bool(args.input_dir):
        raise ValueError("Specify exactly one of --dataset or --input-dir")

    export_images = not args.no_images

    if args.dataset:
        _unpack_one_file(
            args.dataset,
            args.demo_key,
            args.output_root,
            export_images,
            args.fps,
            args.save_frames,
        )
        return

    input_dir = Path(args.input_dir)
    datasets = sorted(input_dir.glob("*.hdf5"))
    if not datasets:
        raise ValueError(f"No .hdf5 files found in {input_dir}")

    for dataset_path in tqdm(datasets, desc="hdf5 files"):
        _unpack_one_file(
            str(dataset_path),
            args.demo_key,
            args.output_root,
            export_images,
            args.fps,
            args.save_frames,
        )


if __name__ == "__main__":
    main()
