import pickle
from logging import getLogger

import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor
import nvtx

logger = getLogger(__name__)


class IPCTensor:
    """包装 GPU tensor，pickle 时只序列化 IPC 句柄，避免 GPU→CPU 拷贝。

    如果 CUDA IPC 句柄获取失败（例如 tensor 底层 storage 来自 NCCL
    shared-memory 或不支持 cudaIpcGetMemHandle 的内存池），自动回退到
    CPU 拷贝以保证可用性。
    """

    def __init__(self, tensor: torch.Tensor):
        self.device = tensor.device
        self.is_cuda = tensor.is_cuda
        self.ipc_handle = None
        self._fallback_cpu: torch.Tensor | None = None  # IPC 失败时的 CPU 备份

        if not self.is_cuda:
            self._tensor = tensor
            return

        # 1. 同步 CUDA stream，确保所有 pending kernel 已完成
        try:
            torch.cuda.synchronize(tensor.device)
        except Exception:
            pass

        # 2. clone 创建完全独立的 CUDA allocation。
        #    原始 tensor 通常来自 hidden_states[start:end]，与模型输出
        #    共享 storage。在 TP 场景下该 storage 可能分配自不支持 IPC
        #    导出的内存池（如 NCCL 共享内存），clone 后获得常规 cudaMalloc
        #    分配的内存，确保 cudaIpcGetMemHandle 能正常工作。
        with nvtx.annotate(f"IPC tensor Clone {tensor}"):
            tensor = tensor.clone().detach()

        # 3. 获取 IPC 句柄；失败则回退到 CPU 拷贝
        try:
            _, self.ipc_handle = reduce_tensor(tensor)
        except RuntimeError:
            logger.warning(
                "reduce_tensor failed on %s, falling back to CPU copy. "
                "Performance will degrade slightly but correctness is preserved.",
                tensor.device,
            )
            nvtx.mark("IPC Tensor RuntimeError")
            self.is_cuda = False
            self._fallback_cpu = tensor.cpu().clone().detach()
            return

        # 持有原始 tensor 引用，防止 GC 提前释放 IPC 内存
        nvtx.mark(f"{tensor} is {self.is_cuda}")
        self._tensor = tensor
    @nvtx.annotate("IPC Tensor reduce")
    def __reduce__(self):
        if self._fallback_cpu is not None:
            return (_rebuild_cpu_tensor_to_device,
                    (self._fallback_cpu, self.device))
        if self.is_cuda and self.ipc_handle is not None:
            return (rebuild_cuda_tensor, self.ipc_handle)
        # 非 CUDA tensor
        return self._tensor.__reduce_ex__(pickle.DEFAULT_PROTOCOL)
    def __repr__(self):
        return f"(IPCTensor) {self._tensor} + {self.is_cuda} + {self._fallback_cpu}"
        pass
    def wrap_cuda_tensors(obj):
        """递归遍历 dict（及嵌套 list/tuple），将所有 CUDA tensor 包装为 IPCTensor。

        非 CUDA tensor、非 tensor 对象原样保留。
        """
        if isinstance(obj, dict):
            return {k: IPCTensor.wrap_cuda_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [IPCTensor.wrap_cuda_tensors(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(IPCTensor.wrap_cuda_tensors(v) for v in obj)
        elif isinstance(obj, torch.Tensor) and obj.is_cuda:
            # logger.info(f"warp {obj} to IPCTensor")
            return IPCTensor(obj)
        else:
            return obj


def _rebuild_cpu_tensor_to_device(
    cpu_tensor: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """从 CPU 备份重建 tensor 并移回原 GPU。"""
    return cpu_tensor.to(device)
