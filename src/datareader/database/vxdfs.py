import clickhouse_connect
from clickhouse_connect.driver.exceptions import OperationalError
import asyncio
from typing import Optional
import logging

logger = logging.getLogger('vxdfs_alg')
from config import VxdfsConfig

class _VXDFS:
    """
    使用信号量控制并发，并具备自动重连功能
    """
    def __init__(self, client_kwargs,max_concurrent_queries=20):
        # 保存创建客户端所需的参数，用于重建
        self.client_kwargs = client_kwargs
        # 创建初始客户端
        #self.client = self._create_client()
        self.client = None
        self.semaphore = asyncio.Semaphore(max_concurrent_queries)
        self._lock = asyncio.Lock()  # 用于保护重建客户端的操作

    def _create_client(self):
        """创建一个新的ClickHouse客户端"""
        logger.info("正在创建新的ClickHouse客户端连接...")
        # 这里可以加入一些关键的连接参数
        return clickhouse_connect.get_client(**self.client_kwargs)

    async def _ensure_client(self):
        """
        确保当前client是可用的。如果不可用，则重建。
        这是一个简单的健康检查，可以根据需要扩展。
        """
        try:
            # 尝试执行一个极快的查询来测试连接
            # 注意：这里使用run_in_executor是因为client的方法可能是同步的
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.client.query, 'SELECT 1')
        except (OperationalError, ConnectionError,AttributeError) as e:
            logger.warning(f"检测到连接可能已断开 ({e})，尝试重建...")
            async with self._lock:  # 防止多个任务同时重建
                try:
                    # 再次检查，防止在等待锁的过程中已被其他任务重建
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.client.query, 'SELECT 1')
                except:
                    # 确认连接确实坏了，进行重建
                    try:
                        self.client.close()  # 尝试关闭旧连接
                    except:
                        pass
                    self.client = self._create_client()
                    logger.info("客户端重建成功。")
        return self.client

    async def execute_query(self, query, params=None, retries=1):
        """
        执行查询，包含自动重连逻辑。
        """
        for attempt in range(retries + 1):
            try:
                # 1. 获取信号量，控制并发
                async with self.semaphore:
                    # 2. 确保连接可用（自动重连的关键步骤）
                    client = await self._ensure_client()

                    # 3. 执行实际查询
                    loop = asyncio.get_event_loop()
                    # 根据你的查询类型，可能是 query, query_df, query_np 等
                    result = await loop.run_in_executor(
                        None, 
                        lambda: client.query(query, parameters=params) if params else client.query(query)
                    )
                    return result

            except (OperationalError, NetworkError, ConnectionError) as e:
                logger.error(f"查询执行失败 (尝试 {attempt + 1}/{retries + 1}): {e}")
                if attempt == retries:
                    # 最后一次尝试也失败了，抛出异常
                    raise
                else:
                    # 等待一小段时间再重试，避免立即重试同样失败
                    await asyncio.sleep(0.5 * (2 ** attempt))  # 简单的退避策略
                    # 强制在下一次循环中重建连接（通过_ensure_client）
                    continue
            except Exception as e:
                # 其他非连接性错误，直接抛出，不重试
                logger.error(f"查询执行遇到非连接性错误: {e}")
                raise

    def close(self):
        """关闭客户端连接"""
        if self.client:
            try:
                self.client.close()
                logger.info("ClickHouse客户端已关闭。")
            except Exception as e:
                logger.error(f"关闭客户端时出错: {e}")

vxdfs = _VXDFS(client_kwargs=VxdfsConfig.config)