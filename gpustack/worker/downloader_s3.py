import os
import logging
import shutil
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
        # 兼容带 http 或 https 前缀的 S3 地址以及末尾可能存在的斜杠
        host = host.rstrip('/')
        if host.startswith("http://"):
            host = host[len("http://"):]
            ssl = False
        elif host.startswith("https://"):
            host = host[len("https://"):]
            ssl = True

        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        http_client = urllib3.PoolManager(
            timeout=urllib3.Timeout.DEFAULT_TIMEOUT,
            cert_reqs='CERT_NONE',
            retries=urllib3.Retry(
                total=5,
                backoff_factor=0.2,
                status_forcelist=[500, 502, 503, 504],
            )
        )

        self._s3_client = Minio(
            host,
            access_key=access_key,
            secret_key=secret_key,
            secure=ssl,
            region=region,
            http_client=http_client if ssl else None,
        )
        self._cache_dir = cache_dir or self._default_cache_dir
        self.use_virtual_hosted_style = use_virtual_hosted_style

    def download(
        self,
        s3_path: str,
        cache_dir: Optional[str] = None,
        strip_prefix: str = "datamodel/",
    ) -> str:
        """从S3下载模型文件

        Args:
            s3_path: S3路径,格式为s3://beagle_wind/bucket/path/to/model
            cache_dir: 本地缓存目录

        Returns:
            下载后的本地文件路径
        """
        if not s3_path.startswith("s3://"):
            return s3_path

        s3_path = self.normalize_s3_path(s3_path)

        # 解析S3路径
        base_path = s3_path.removeprefix("s3://")
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
        cache_relative_path = self._cache_relative_path(
            s3_path,
            bucket_name,
            strip_prefix=strip_prefix,
        )
        local_path = os.path.join(
            local_cache,
            cache_relative_path,
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
                objects = list(objects)
                if not objects:
                    raise FileNotFoundError(
                        f"No S3 objects found for {s3_path} with prefix {object_prefix}"
                    )

                matched_local_cache = True
                expected_local_paths = {
                    os.path.join(
                        local_cache,
                        self._cache_relative_path(
                            f"s3://{bucket_name}/{obj.object_name}",
                            bucket_name,
                            strip_prefix=strip_prefix,
                        ),
                    )
                    for obj in objects
                }
                if not is_file and os.path.isdir(local_dir):
                    local_files = {
                        os.path.join(root, file)
                        for root, _, files in os.walk(local_dir)
                        for file in files
                    }
                    extra_local_files = local_files - expected_local_paths
                    if extra_local_files:
                        logger.warning(
                            f"Local cache has files not present in S3, redownloading: {local_dir}"
                        )
                        shutil.rmtree(local_dir)
                        os.makedirs(local_dir, exist_ok=True)
                        matched_local_cache = False

                # 下载每个对象
                for obj in objects:
                    object_relative_path = self._cache_relative_path(
                        f"s3://{bucket_name}/{obj.object_name}",
                        bucket_name,
                        strip_prefix=strip_prefix,
                    )
                    local_object_path = os.path.join(local_cache, object_relative_path)
                    downloaded = self._download_object(
                        bucket_name=bucket_name,
                        object_name=obj.object_name,
                        local_path=local_object_path,
                        total_size=obj.size,
                    )
                    if downloaded:
                        matched_local_cache = False

                if matched_local_cache:
                    logger.info(
                        f"Local cache matched S3 metadata, reuse local path: {local_path if is_file else local_dir}"
                    )
                else:
                    logger.debug(f"已下载 {bucket_name}/{object_prefix}")
                return local_path if is_file else local_dir

            except S3Error as e:
                logger.error(f"S3下载错误: {e}")
                raise

    @staticmethod
    def normalize_s3_path(s3_path: str) -> str:
        # 兼容旧版本的固定前缀 s3://beagle_wind/
        if s3_path.startswith("s3://beagle_wind/"):
            return s3_path.replace("s3://beagle_wind/", "s3://", 1)
        return s3_path

    @classmethod
    def parse_s3_path(cls, s3_path: str) -> tuple[str, str]:
        s3_path = cls.normalize_s3_path(s3_path)
        base_path = s3_path.removeprefix("s3://")
        bucket_name = base_path.split("/")[0]
        object_path = base_path.removeprefix(bucket_name).lstrip("/")
        return bucket_name, object_path

    @staticmethod
    def _cache_relative_path(
        s3_path: str,
        bucket_name: str,
        strip_prefix: str = "datamodel/",
    ) -> str:
        object_path = s3_path.removeprefix(f"s3://{bucket_name}/")
        if strip_prefix and object_path.startswith(strip_prefix):
            object_path = object_path[len(strip_prefix) :]
        return object_path

    def list_file_entries(
        self,
        s3_path: str,
        strip_prefix: str = "",
    ) -> List[FileEntry]:
        bucket_name, object_prefix = self.parse_s3_path(s3_path)
        objects = list(
            self._s3_client.list_objects(
                bucket_name,
                prefix=object_prefix,
                recursive=True,
            )
        )
        if not objects:
            return []

        return [
            FileEntry(
                obj.object_name.removeprefix(strip_prefix),
                obj.size,
            )
            for obj in objects
        ]

    def _download_object(
        self, bucket_name: str, object_name: str, local_path: str, total_size: int
    ) -> bool:
        """分片下载单个S3对象"""
        downloaded_size = 0
        if os.path.exists(local_path):
            downloaded_size = os.path.getsize(local_path)

        if downloaded_size == total_size:
            logger.debug(f"文件 {object_name} 已存在且大小一致,跳过下载")
            return False

        if downloaded_size > total_size:
            logger.warning(
                f"Local file {local_path} is larger than S3 object {object_name}, redownloading"
            )
            downloaded_size = 0

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

        return True

    def get_model_file_size(self, model_instance: ModelInstance) -> List[FileEntry]:
        """获取S3上模型文件的总大小

        Args:
            model_instance: 模型实例对象

        Returns:
            文件总大小(字节)

        Raises:
            S3Error: 当S3操作失败时抛出
        """
        if not model_instance.local_path.startswith("s3://"):
            return None

        try:
            s3_path = model_instance.local_path
            s3_path = self.normalize_s3_path(s3_path)

            # 解析S3路径
            base_path = s3_path.removeprefix("s3://")
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
