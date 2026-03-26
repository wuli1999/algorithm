from typing import Optional
from enum import Enum, unique

@unique
class ErrorCode(Enum):
    # 客户端错误 (4xxx)
    INVALID_PARAMETER = (4001, "请求参数无效")
    RESOURCE_NOT_FOUND = (4004, "请求资源不存在")
    PERMISSION_DENIED = (4003, "权限不足")
    DATA_ERROR = (4002, "数据错误")
    PROCESS_FAILURE = (4003, "处理失败")

    @property
    def code(self):
        """获取错误码数字"""
        return self.value[0]

    @property
    def message(self):
        """获取错误描述信息"""
        return self.value[1]

class BusinessValidationError(Exception):
    """自定义业务验证异常类"""
    def __init__(self, message: str, error_code: ErrorCode):
        super().__init__(message)
        self.message = message
        self.error_code = error_code.code
        self.error_type = error_code.message
