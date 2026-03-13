import os
import logging
from pathlib import Path
from typing import List, Optional
from minio import Minio
from minio.error import S3Error
from filelock import FileLock
from tqdm import tqdm

from gpustack.schemas.models import (
    ModelInstance,
)
from gpustack.utils.hub import (
    FileEntry,
)

logger = logging.getLogger(__name__)


class S3Downloader:
    _default_cache_dir = "/var/lib/gpustack/cache/beagle"

    def __init__(
        self,
        host: str,
        access_key: str,
        secret_key: str,
        ssl: bool = False,
        use_virtual_hosted_style: bool = True,
        cache_dir: Optional[str] = None,
        region: str = "",
    ):
        self._s3_client = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=ssl,
            region=region,
        )
        self._cache_dir = cache_dir or self._default_cache_dir
        self.use_virtual_hosted_style = use_virtual_hosted_style

    def download(
        self,
        s3_path: str,
        cache_dir: Optional[str] = None,
    ) -> str:
        """从S3下载模型文件

        Args:
            s3_path: S3路径,格式为s3://beagle_wind/bucket/path/to/model
            cache_dir: 本地缓存目录

        Returns:
            下载后的本地文件路径
        """
        if not s3_path.startswith("s3://beagle_wind/"):
            return s3_path

        # 解析S3路径
        # s3://beagle_wind/bd-wind/datamodel/4c3c6c88-912c-48da-910c-fea84da1fedc/v1/qwen2.5-3b-instruct-q8_0.gguf
        # 或 s3://beagle_wind/bd-wind/datamodel/Qwen/Qwen3.5-35B-A3B-FP8 (目录)
        base_path = s3_path.removeprefix("s3://beagle_wind/")
        bucket_name = base_path.split("/")[0]
        if self.use_virtual_hosted_style:
            base_path = base_path.removeprefix(bucket_name)
        
        # 判断是文件还是目录
        # 如果路径以文件扩展名结尾，认为是文件；否则认为是目录
        is_file = any(s3_path.endswith(ext) for ext in ['.gguf', '.safetensors', '.bin', '.pth', '.pt'])
        
        if is_file:
            # 原有逻辑：下载单个文件及其分片
            object_prefix = "/".join(base_path.split("/")[1:-1])  # 去掉最后的文件名部分
        else:
            # 新逻辑：下载整个目录
            object_prefix = "/".join(base_path.split("/")[1:])  # 保留完整路径作为前缀

        # 准备本地缓存路径
        local_cache = cache_dir or self._cache_dir
        local_path = os.path.join(
            local_cache, s3_path.removeprefix("s3://beagle_wind/"+bucket_name+"/datamodel/")
        )
        local_dir = os.path.dirname(local_path) if is_file else local_path
        if not os.path.exists(local_dir):  # 创建父目录
            os.makedirs(local_dir)

        # 使用文件锁避免并发下载
        lock_filename = local_dir + ".lock"
        logger.debug("获取文件锁")

        with FileLock(lock_filename):
            try:
                # 列出所有对象
                objects = self._s3_client.list_objects(
                    bucket_name, prefix=object_prefix, recursive=True
                )

                # 下载每个对象
                for obj in objects:
                    self._download_object(
                        bucket_name=bucket_name,
                        object_name=obj.object_name,
                        local_path=os.path.join(
                            local_cache, obj.object_name.removeprefix("datamodel/")
                        ),
                        total_size=obj.size,
                    )

                logger.debug(f"已下载 {bucket_name}/{object_prefix}")
                return local_path if is_file else local_dir

            except S3Error as e:
                logger.error(f"S3下载错误: {e}")
                raise

    def _download_object(
        self, bucket_name: str, object_name: str, local_path: str, total_size: int
    ):
        """分片下载单个S3对象"""
        downloaded_size = 0
        if os.path.exists(local_path):
            downloaded_size = os.path.getsize(local_path)

        if downloaded_size >= total_size:
            logger.debug(f"文件 {object_name} 已存在,跳过下载")
            return

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        # 如果是新下载，先清空文件
        if downloaded_size == 0:
            open(local_path, 'wb').close()

        part_size = 10 * 1024 * 1024  # 10MB
        start_byte = downloaded_size

        with tqdm(
            total=total_size,
            initial=downloaded_size,
            desc=object_name,
            unit='B',
            unit_scale=True,
        ) as pbar:
            while start_byte < total_size:
                response = self._s3_client.get_object(
                    bucket_name,
                    object_name,
                    offset=start_byte,
                    length=min(part_size, total_size - start_byte),
                )

                data = response.read()

                # 始终使用追加模式
                with open(local_path, "ab") as f:
                    f.write(data)

                chunk_size = len(data)
                start_byte += chunk_size
                pbar.update(chunk_size)

    def get_model_file_size(self, model_instance: ModelInstance) -> List[FileEntry]:
        """获取S3上模型文件的总大小

        Args:
            model_instance: 模型实例对象

        Returns:
            文件总大小(字节)

        Raises:
            S3Error: 当S3操作失败时抛出
        """
        if not model_instance.local_path.startswith("s3://beagle_wind/"):
            return None

        try:
            # 解析S3路径
            base_path = model_instance.local_path.removeprefix("s3://beagle_wind/")
            bucket_name = base_path.split("/")[0]
            if self.use_virtual_hosted_style:
                base_path = base_path.removeprefix(bucket_name)
            object_prefix = "/".join(base_path.split("/")[1:-1])  # 去掉最后的文件名部分

            # 列出所有对象
            objects = self._s3_client.list_objects(
                bucket_name, prefix=object_prefix, recursive=True
            )
            # 准备本地缓存路径
            local_cache = self._cache_dir
            file_list = [
                FileEntry(
                    os.path.join(local_cache, f.object_name.removeprefix("datamodel/")),
                    f.size,
                )
                for f in objects
            ]
            # 计算所有对象的总大小
            total_size = sum(obj.size for obj in objects)

            logger.debug(
                f"S3路径 {model_instance.local_path} 的总大小: {total_size} 字节"
            )
            return file_list

        except S3Error as e:
            logger.error(f"获取S3文件大小失败: {e}")
            raise
