# LIBERO 数据集下载、查看、训练与测试教程

本教程以 `libero_object` 为例，覆盖官方数据下载、`hdf5` 内容查看、本地生成数据，以及训练和测试。

## 0. 环境准备

```bash
pip install -r requirements.txt
pip install -e .
export DATA_ROOT=libero_data
mkdir -p $DATA_ROOT
```

## 1. 下载官方子集

```bash
python benchmark_scripts/download_libero_datasets.py \
  --datasets libero_object \
  --download-dir $DATA_ROOT/official \
  --use-huggingface
```

## 2. 查看 `hdf5` 里有什么

官方 `*_demo.hdf5` 已经是训练可用格式，不需要额外解码。若要先看文件结构，可运行：

```bash
python scripts/get_dataset_info.py \
  --dataset $DATA_ROOT/official/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

如果你想把一个目录里的所有 `hdf5` 都展开成普通文件目录，可运行：

```bash
python scripts/unpack_hdf5_dataset.py \
  --input-dir $DATA_ROOT/official/libero_object
```

这个命令会对目录下每个 `*.hdf5` 单独解包，并把结果放在原文件旁边的同名目录里。运行时会显示文件级和视频写出进度条。例如：

```text
$DATA_ROOT/official/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
-> $DATA_ROOT/official/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo/
```

如果你只想解一个文件：

```bash
python scripts/unpack_hdf5_dataset.py \
  --dataset $DATA_ROOT/official/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5
```

如果你只想看其中一条轨迹，例如 `demo_0`：

```bash
python scripts/unpack_hdf5_dataset.py \
  --dataset $DATA_ROOT/official/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --demo-key demo_0
```

如果你不想导出视频，只保留 `json` 和 `npy`：

```bash
python scripts/unpack_hdf5_dataset.py \
  --input-dir $DATA_ROOT/official/libero_object \
  --no-images
```

如果你还想额外保留逐帧 `png`，显式加上：

```bash
python scripts/unpack_hdf5_dataset.py \
  --input-dir $DATA_ROOT/official/libero_object \
  --save-frames
```

解包后的内容通常包括：
- `attrs.json`：任务和环境元数据
- `*.npy`：动作、状态、奖励、图像张量
- `*_frames/video.mp4`：由图像张量还原出的完整视频
- `*_frames/*.png`：仅在使用 `--save-frames` 时导出

一个典型目录结构如下：

```text
pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo/
  attrs.json
  data/
    attrs.json
    demo_1/
      attrs.json
      actions.npy
      actions.meta.json
      states.npy
      rewards.npy
      dones.npy
      obs/
        agentview_rgb.npy
        agentview_rgb.meta.json
        agentview_rgb_frames/
          video.mp4
```

## 3. 本地生成数据

有图形界面时可人工采集：

```bash
python scripts/collect_demonstration.py \
  --device keyboard \
  --num-demonstration 50 \
  --directory $DATA_ROOT/raw_demos \
  --bddl-file libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl
```

无图形界面时可用已训练策略无头采集：

```bash
python scripts/headless_collect_from_policy.py \
  --benchmark libero_object \
  --task-id 0 \
  --algo base \
  --policy bc_rnn_policy \
  --seed 0 \
  --load_task 9 \
  --device-id 0 \
  --num-demos 50 \
  --output $DATA_ROOT/raw_policy/libero_object_task0_demo.hdf5
```

原始 `demo.hdf5` 需要转换成训练格式：

```bash
python scripts/create_dataset.py \
  --demo-file <path/to/demo.hdf5> \
  --use-camera-obs
```

## 4. 训练与测试

用官方下载子集训练：

```bash
python libero/lifelong/main.py \
  benchmark_name=LIBERO_OBJECT \
  policy=bc_rnn_policy \
  lifelong=base \
  seed=0 \
  folder=$DATA_ROOT/official
```

测试：

```bash
python libero/lifelong/evaluate.py \
  --benchmark libero_object \
  --task_id 0 \
  --algo base \
  --policy bc_rnn_policy \
  --seed 0 \
  --load_task 9 \
  --device_id 0
```
