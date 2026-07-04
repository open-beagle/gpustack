import logging


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

        for instance in stale_instances:
            logger.info(
                f"Deleting stale model instance {instance.name} "
                f"assigned to restarting worker {worker_name}."
            )
            clientset.model_instances.delete(id=instance.id)
    except Exception as e:
        logger.error(
            f"Failed to cleanup stale model instances for worker {worker_name}: {e}"
        )
