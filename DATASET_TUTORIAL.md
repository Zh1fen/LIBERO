# LIBERO 子集数据集下载、生成、训练与测试教程（含可视化）

本教程以 `libero_object` 为例，覆盖：
1. 下载官方子集
2. 本地生成同子集（含无图形方案）
3. 训练与评测
4. 每个阶段的视频可视化

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

## 2. 本地生成子集

你可以选择两种方式。

### 2.1 方式 A：人工采集（需要图形界面）

```bash
python scripts/collect_demonstration.py \
  --device keyboard \
  --num-demonstration 50 \
  --directory $DATA_ROOT/raw_demos \
  --bddl-file libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl
```

### 2.2 方式 B：无图形采集（方案三，策略 rollout）

先保证你已有可用 checkpoint（例如先用官方数据训练出一个策略），然后在无 GUI 环境执行：

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

要生成完整 `libero_object` 子集，请对 `task-id 0..9` 分别执行一次。

### 2.3 转训练格式（A/B 两种方式都适用）

```bash
python scripts/create_dataset.py \
  --demo-file <path/to/demo.hdf5> \
  --use-camera-obs
```

### 2.4 检查数据

```bash
python scripts/get_dataset_info.py --dataset <path/to/one_task_demo.hdf5>
```

## 3. 训练

### 3.1 用官方下载子集训练

```bash
python libero/lifelong/main.py \
  benchmark_name=LIBERO_OBJECT \
  policy=bc_rnn_policy \
  lifelong=base \
  seed=0 \
  folder=$DATA_ROOT/official
```

### 3.2 用本地生成子集训练

```bash
python libero/lifelong/main.py \
  benchmark_name=LIBERO_OBJECT \
  policy=bc_rnn_policy \
  lifelong=base \
  seed=0 \
  folder=<你的本地数据根目录>
```

## 4. 评测（测试）

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

## 5. 每个阶段视频可视化

仓库已提供脚本：`scripts/export_demo_video.py`，可直接把 hdf5 轨迹导出为 mp4。

### 5.1 看“官方下载数据”的视频

```bash
python scripts/export_demo_video.py \
  --dataset $DATA_ROOT/official/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo.hdf5 \
  --demo-key demo_0 \
  --output $DATA_ROOT/videos/official_demo0.mp4
```

### 5.2 看“本地 raw demo.hdf5（人工或策略采集）”的回放视频

```bash
python scripts/export_demo_video.py \
  --dataset <path/to/raw/demo.hdf5> \
  --demo-key demo_1 \
  --output $DATA_ROOT/videos/raw_demo1.mp4
```

### 5.3 看“转换后训练格式数据”的视频

```bash
python scripts/export_demo_video.py \
  --dataset <path/to/converted_task_demo.hdf5> \
  --demo-key demo_0 \
  --output $DATA_ROOT/videos/converted_demo0.mp4
```

### 5.4 看“测试阶段模型 rollout 视频”

```bash
python libero/lifelong/evaluate.py \
  --benchmark libero_object \
  --task_id 0 \
  --algo base \
  --policy bc_rnn_policy \
  --seed 0 \
  --load_task 9 \
  --device_id 0 \
  --save-videos
```

视频输出到：`<experiment_dir>_saved/*_videos/video.mp4`。
