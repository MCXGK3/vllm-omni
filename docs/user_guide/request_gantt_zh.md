# 请求执行甘特图使用说明

该功能用于记录一次 vLLM-Omni 服务中的 Thinker、Talker 请求执行过程，并在服务结束后生成请求级甘特图。图中可以区分首次预填充、分块预填充、解码以及同一请求在一次 forward 中同时跨越预填充和解码边界的情况。

## 1. 开启采集

启动服务前设置以下环境变量：

```bash
export VLLM_OMNI_REQUEST_GANTT=1
export VLLM_OMNI_REQUEST_GANTT_DIR=/home/request_gantt_trace
export VLLM_OMNI_REQUEST_GANTT_SESSION=gantt_001
export VLLM_OMNI_REQUEST_GANTT_TIMING_MODE=cuda_event
```

然后使用原来的方式启动服务和测试：

```bash
cd /home
bash scripts/start_vllm_omni.sh scripts/bench.sh
```

变量含义如下：

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `VLLM_OMNI_REQUEST_GANTT` | `0` | 是否开启请求甘特图采集 |
| `VLLM_OMNI_REQUEST_GANTT_DIR` | `/home/request_gantt_trace` | 采集文件根目录 |
| `VLLM_OMNI_REQUEST_GANTT_SESSION` | `default` | 本次服务的会话名称 |
| `VLLM_OMNI_REQUEST_GANTT_TIMING_MODE` | `sync` | 计时模式：`sync`、`cuda_event` 或 `cpu` |
| `VLLM_OMNI_REQUEST_GANTT_SYNC_GPU` | `1` | 兼容旧配置；未设置 timing mode 时，`1` 对应 `sync`，`0` 对应 `cpu` |

建议每次测试使用不同的 session 名称，避免不同服务运行的数据混在一起。

## 2. 数据文件

每个 stage 的 TP rank 0 写入独立 JSONL 文件，不同 stage 不会争用同一个输出文件。其余 TP rank 不重复记录相同 batch。例如：

```text
/home/request_gantt_trace/gantt_001/
├── stage_0_thinker_pid_1234.jsonl
└── stage_1_talker_pid_1250.jsonl
```

每条记录代表一次完整 forward batch，主要字段包括：

- `batch_id`：本次 forward 的唯一编号；
- `stage_id`、`stage_name`：Thinker/Talker 阶段；
- `start_ns`、`end_ns`：基于单机 monotonic clock 的起止时间；
- `total_scheduled_tokens`：batch 总 token 数；
- `requests`：batch 内各请求的 token 和执行阶段信息；
- `execution`：CUDA Graph、padding、ubatch 等执行元数据。

Thinker 和 Talker 使用 `global_request_id` 对齐同一个服务请求。

## 3. 生成甘特图

在仓库根目录执行：

```bash
cd /home/codes/qyf_vllm_omni/vllm-omni

python3 experiments/scripts/render_request_gantt.py \
  /home/request_gantt_trace/gantt_001 \
  --html /home/request_gantt_trace/gantt_001/请求执行甘特图.html \
  --png /home/request_gantt_trace/gantt_001/请求执行甘特图.png \
  --include-gaps
```

脚本会递归读取目录内所有 JSONL，并生成：

- HTML：自包含 SVG，鼠标悬停可查看 batch、token 和时长；
- PNG：适合直接放入实验报告。

如果运行环境没有安装中文字体，PNG 的标题和图例会自动使用英文，避免出现缺失字形；HTML 仍由浏览器使用系统字体显示中文。

只查看指定请求：

```bash
python3 experiments/scripts/render_request_gantt.py \
  /home/request_gantt_trace/gantt_001 \
  --request-id chatcmpl-xxxxxxxx \
  --html request.html \
  --png request.png
```

当 decode iteration 很多时，可以合并间隔很小的连续 decode 条带：

```bash
python3 experiments/scripts/render_request_gantt.py \
  /home/request_gantt_trace/gantt_001 \
  --merge-decode-gap-ms 2.0
```

## 4. 图像含义

| 颜色 | 含义 |
|---|---|
| 蓝色 | 首次预填充 |
| 橙色 | chunked prefill |
| 绿色 | decode |
| 紫色 | 同一请求在一次 forward 中同时包含 prefill 和 decode token |
| 灰色 | 相邻两次 forward 之间没有执行该请求的时间间隔 |

纵轴每一行由 `global_request_id / stage` 组成，横轴是相对本次 trace 第一条记录的时间。

同一个 batch 内的请求共享一次 GPU forward，因此图中这些请求具有相同的起止时间。甘特图不会把 batch 总时间错误拆分成多个独立的请求执行时间。

每个真实执行批次还会得到一个短编号，例如 `S0-B02`：

- `S0` 表示 stage 0（Thinker），`S1` 表示 stage 1（Talker）。
- `B02` 表示该阶段按开始时间排序的第 2 个 forward 批次。
- 不同请求行中具有相同短编号和边框颜色的色块属于同一批次。
- 很窄的解码色块可能不直接绘制文字，以免互相覆盖；HTML 中悬停仍可查看短编号和完整 `batch_id`。

## 5. 精度与开销

支持三种计时模式。

### `sync`

```bash
export VLLM_OMNI_REQUEST_GANTT_TIMING_MODE=sync
```

在 forward 前后调用 `torch.cuda.synchronize()`。记录的是 GPU 已完成的 forward 时间，但会改变异步流水、调度节奏和阶段间重叠，只建议用于受控实验。

### `cuda_event`

```bash
export VLLM_OMNI_REQUEST_GANTT_TIMING_MODE=cuda_event
```

在当前 CUDA stream 上记录 start/end Event，不在 forward 热路径调用 `synchronize()`。后续 forward 会查询已经完成的 Event，并追加 `forward_timing_update` 记录；worker 退出时会尽力清空最后一批 Event。

该模式的 `duration_ms` 来自 CUDA Event，能够反映 GPU stream 上两 Event 之间的执行时间。甘特图的绝对起点仍以 CPU 提交 start Event 的 `perf_counter` 时间为锚点，因此当 CUDA stream 前方存在较长排队时，跨 stage 的绝对位置可能有少量偏差。

这是兼顾测量精度和运行扰动的推荐模式。

### `cpu`

```bash
export VLLM_OMNI_REQUEST_GANTT_TIMING_MODE=cpu
```

只记录 CPU 调用 `_model_forward()` 的提交区间，开销最低，但不能直接作为 GPU 执行时间。

旧开关仍然可用：

```bash
export VLLM_OMNI_REQUEST_GANTT_SYNC_GPU=0
```

如果同时设置了 `VLLM_OMNI_PREFILL_TIMING=1`，为了保持原有计时语义，会强制使用 `sync` 模式。

## 6. 与原有 prefill timing 的关系

原有环境变量仍然可以继续使用：

```bash
export VLLM_OMNI_PREFILL_TIMING=1
export VLLM_OMNI_PREFILL_TIMING_PATH=/home/prefill_timing.jsonl
```

甘特图采集和原有 prefill timing 可以同时开启。前者按 session、stage 和 PID 分文件，后者维持原来的单文件格式，已有实验脚本不受影响。

## 7. 当前边界

- 当前版本记录 Thinker/Talker 的模型 forward，不包含网络客户端到达 API Server 的时间。
- 灰色间隔表示请求在相邻 forward 之间未执行，不严格等同于 scheduler queue time。
- batch 内请求并行执行，无法仅从 batch forward 中测得单个请求独占的 GPU 时间。
- 如果需要进一步展示 UniIPC 发送/接收，可在 connector 的 send/recv 路径中加入相同格式的 span 事件。
