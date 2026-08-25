import glob
import logging
import os

from gpustack.api.exceptions import HTTPException
from gpustack.schemas.model_files import ModelFileStateEnum


logger = logging.getLogger(__name__)


def reconcile_ready_model_files(clientset, worker_id: int, worker_name: str):
    """标记当前 Worker 上已丢失本地源路径的 READY 模型文件。

    仅核对服务端冻结的 ``resolved_paths``，不访问其他 Worker，也不会删除文件。
    """
    try:
        model_files = []
        page = 1
        while True:
            result = clientset.model_files.list(
                params={
                    "worker_id": worker_id,
                    "state": ModelFileStateEnum.READY.value,
                    "page": page,
                    "perPage": 100,
                }
            )
            if not result or not result.items:
                break
            model_files.extend(result.items)
            page += 1
    except Exception:
        logger.exception("获取 worker %s 的 READY 模型文件失败。", worker_name)
        return

    for model_file in model_files:
        if _resolved_paths_exist(model_file.resolved_paths):
            continue
        try:
            clientset.model_storage_sync_tasks.mark_model_file_source_missing(
                model_file.id,
                model_file.updated_at,
            )
        except HTTPException as exc:
            if exc.status_code == 404:
                continue
            logger.exception(
                "更新 worker %s 的遗留模型文件状态失败: %s", worker_name, exc
            )
        except Exception as exc:
            logger.exception(
                "更新 worker %s 的遗留模型文件状态失败: %s", worker_name, exc
            )


def _resolved_paths_exist(resolved_paths) -> bool:
    """所有冻结路径均存在；glob 至少匹配一个路径才视为存在。"""
    return bool(resolved_paths) and all(
        (
            bool(glob.glob(path, recursive=True))
            if glob.has_magic(path)
            else os.path.exists(path)
        )
        for path in resolved_paths
    )


def cleanup_stale_model_instances(clientset, worker_id: int, worker_name: str):
    """清理 worker 重启后遗留的实例记录，让模型控制器重新补齐副本。"""
    try:
        stale_instances = []
        page = 1
        while True:
            instances = clientset.model_instances.list(
                params={
                    "worker_id": worker_id,
                    "page": page,
                    "perPage": 100,
                }
            )
            if not instances or not instances.items:
                break

            stale_instances.extend(instances.items)
            page += 1

    except Exception:
        logger.exception(f"获取 worker {worker_name} 的遗留模型实例失败。")
        return

    for instance in stale_instances:
        logger.info(
            f"正在删除已重启 worker {worker_name} 的遗留模型实例 {instance.name}。"
        )
        try:
            clientset.model_instances.delete(id=instance.id)
        except HTTPException as e:
            if e.status_code == 404:
                logger.info(f"遗留模型实例 {instance.name} 已被其他清理操作删除。")
                continue
            logger.exception(
                f"删除 worker {worker_name} 的遗留模型实例 {instance.name} 失败: {e}"
            )
        except Exception as e:
            logger.exception(
                f"删除 worker {worker_name} 的遗留模型实例 {instance.name} 失败: {e}"
            )
