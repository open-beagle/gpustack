# vox-box 音频后端说明

## 背景

vox-box 是 GPUStack 的音频推理后端，支持 TTS（文本转语音）和 ASR（语音识别）等功能。

由于 vox-box 锁死依赖 `aiohttp==3.11.2`，与 vllm 要求的 `aiohttp>=3.13.3` 冲突，无法在同一个 Python 环境中共存。因此 vox-box 已从 `pyproject.toml` 的直接依赖中移除，改为运行时通过 pipx 按需安装到独立虚拟环境。

## 运行时行为

当用户部署音频模型时，GPUStack 会自动通过 pipx 将 vox-box 安装到独立环境（`/var/lib/gpustack/pipx`），然后以子进程方式启动 `vox-box start` 服务。

## 注意事项

1. 运行音频模型的节点需要能访问 PyPI（或配置的镜像源）以便 pipx 安装 vox-box
2. 如果 Docker 容器映射了本地 `/var/lib/gpustack` 目录，首次部署音频模型时会触发 vox-box 安装，后续不再重复
3. 离线环境需要提前在节点上手动安装 vox-box：

```bash
# 确保 pipx 已安装
pip install pipx

# 安装 vox-box 到独立环境
PIPX_HOME=/var/lib/gpustack/pipx \
PIPX_BIN_DIR=/var/lib/gpustack/bin \
pipx install vox-box==0.0.21
```

4. 音频模型的资源评估功能（显存/内存预估）在当前版本中不可用，部署时会跳过精确资源评估

## 受影响的功能

- TTS（文本转语音）：正常工作，运行时自动安装
- ASR（语音识别）：正常工作，运行时自动安装
- 音频模型资源评估：不可用（import vox_box 会失败），不影响模型部署，仅影响调度精度

## 恢复条件

当 vox-box 上游修复 aiohttp 版本约束（不再锁死 3.11.2）后，可以将其重新加入 pyproject.toml：

```toml
vox-box = {version = ">=0.0.22", optional = true}
```
