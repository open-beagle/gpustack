import logging

from gpustack.api.exceptions import HTTPException


logger = logging.getLogger(__name__)


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
