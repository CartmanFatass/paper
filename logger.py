import logging
import os
import sys
from datetime import datetime
import multiprocessing
from logging.handlers import QueueHandler, QueueListener
import pickle  # 用于序列化日志记录

# 全局队列和监听器
_log_queue = None
_queue_listener = None

def _setup_log_process(queue, log_file_path, file_level, console_level):
    """配置负责将日志写入文件的监听器"""
    # 创建和配置处理器
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8', mode='a')  # 附加模式
    file_handler.setLevel(file_level)
    
    # 创建控制台处理器并设置级别
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    
    # 创建格式化器，使其更明显地显示进程名称和日志记录器名称
    formatter = logging.Formatter('%(asctime)s - [%(processName)s] - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 创建并启动队列监听器
    # respect_handler_level=True确保处理器自己的级别过滤生效
    listener = QueueListener(queue, file_handler, console_handler, respect_handler_level=True)
    listener.start()
    
    return listener

def init_multiproc_logging(log_dir='logs', log_file=None, file_level=logging.INFO, console_level=logging.WARNING):
    """初始化多进程日志系统"""
    global _log_queue, _queue_listener, _mp_manager
    
    # 先关闭已有的日志监听器，避免重复
    try:
        if _queue_listener:
            _queue_listener.stop()
    except (AttributeError, Exception) as e:
        print(f"[logger] 警告: 停止现有队列监听器时出错: {e}")
    finally:
        _queue_listener = None
    
    # 创建日志目录
    os.makedirs(log_dir, exist_ok=True)
    
    # 如果未提供日志文件名，则使用时间戳生成
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"multiproc_{timestamp}.log"
    
    log_file_path = os.path.join(log_dir, log_file)
    
    # 创建全局Manager，确保它在模块级别持久存在
    global _mp_manager
    try:
        if _mp_manager is None:
            _mp_manager = multiprocessing.Manager()
            print(f"[logger] 创建新Manager: {_mp_manager}")
        else:
            print(f"[logger] 使用现有Manager: {_mp_manager}")
        
        # 使用全局Manager创建队列
        _log_queue = _mp_manager.Queue()
    except (AttributeError, Exception) as e:
        print(f"[logger] 错误: Manager创建或获取队列时出错: {e}")
        # 尝试重新创建Manager
        try:
            _mp_manager = multiprocessing.Manager()
            print(f"[logger] 重新创建Manager: {_mp_manager}")
            _log_queue = _mp_manager.Queue()
        except Exception as e2:
            print(f"[logger] 严重错误: 无法创建Manager: {e2}")
            raise
    print(f"[logger] 创建队列: {_log_queue}")
    
    # 创建并启动监听器，负责将队列中的日志记录写入文件
    _queue_listener = _setup_log_process(_log_queue, log_file_path, file_level, console_level)
    
    # 获取并配置根日志记录器
    root = logging.getLogger()
    
    # 设置根日志记录器级别
    root.setLevel(logging.DEBUG)  # 设置为最低级别，让处理器决定过滤
    
    # 清除任何现有的处理器
    while root.handlers:
        root.removeHandler(root.handlers[0])
    
    # 添加队列处理器
    queue_handler = QueueHandler(_log_queue)
    root.addHandler(queue_handler)
    
    # 配置默认的main_logger
    global main_logger
    if main_logger:
        # 清除现有处理器
        while main_logger.handlers:
            main_logger.removeHandler(main_logger.handlers[0])
        
        # 设置级别并添加处理器
        main_logger.setLevel(logging.DEBUG)  # 设置为最低级别
        main_logger.propagate = False  # 阻止日志传递给根记录器，避免重复
        main_logger.addHandler(queue_handler)
    
    return root

def shutdown_logging():
    """关闭日志系统，确保所有日志都被记录"""
    global _queue_listener, _log_queue, _mp_manager
    try:
        if _queue_listener:
            _queue_listener.stop()
    except (AttributeError, Exception) as e:
        print(f"[logger] 警告: 关闭日志系统时出错: {e}")
    finally:
        _queue_listener = None
        _log_queue = None
        
        # 尝试关闭Manager并释放资源
        try:
            if _mp_manager:
                _mp_manager.shutdown()
        except (AttributeError, Exception) as e:
            print(f"[logger] 警告: 关闭Manager时出错: {e}")
        finally:
            _mp_manager = None

def get_logger(name):
    """获取配置好的子日志记录器"""
    logger = logging.getLogger(name)
    
    # 确保此日志记录器使用队列处理器
    if _log_queue:
        # 清除任何现有的处理器
        while logger.handlers:
            logger.removeHandler(logger.handlers[0])
        
        # 添加队列处理器
        queue_handler = QueueHandler(_log_queue)
        logger.addHandler(queue_handler)
        
        # 确保级别设置正确（不要过滤任何消息）
        logger.setLevel(logging.DEBUG)
        
        # 重要：设置propagate=False阻止日志向上传递到根记录器
        # 这能防止日志被记录两次
        logger.propagate = False
    
    return logger

# 全局Manager对象，确保在所有进程间共享
_mp_manager = None

# 直接记录到队列的函数，用于处理无法使用标准日志记录器的情况
def log_directly(level, name, message):
    """直接将日志消息添加到队列，绕过日志记录器"""
    global _log_queue
    try:
        if _log_queue:
            # 创建一个记录
            record = logging.LogRecord(
                name=name,
                level=level,
                pathname="",
                lineno=0,
                msg=message,
                args=(),
                exc_info=None
            )
            # 将记录添加到队列
            _log_queue.put_nowait(record)
            return True
        else:
            print(f"[log_directly] 警告: _log_queue 不可用")
            return False
    except Exception as e:
        print(f"[log_directly] 错误: {e}")
        return False

# 日志级别常量，用于用户友好的参数选择
LOG_LEVELS = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warning': logging.WARNING,
    'error': logging.ERROR,
    'critical': logging.CRITICAL
}

def set_log_level(level_name, console_level_name=None):
    """设置日志级别"""
    level = LOG_LEVELS.get(level_name.lower(), logging.INFO)
    console_level = LOG_LEVELS.get(console_level_name.lower(), level) if console_level_name else level
    
    # 更新根日志记录器的级别
    root_logger = logging.getLogger()
    root_logger.setLevel(min(level, console_level))

# 保留原有的setup_logger函数用于向后兼容
def setup_logger(name, log_dir='debug', log_file=None, level=logging.INFO, console_level=logging.WARNING):
    """单进程日志设置函数（保留用于向后兼容）"""
    os.makedirs(log_dir, exist_ok=True)
    
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"{name}_{timestamp}.log"
    
    log_file_path = os.path.join(log_dir, log_file)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(level)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    
    return logger

# 创建一个兼容的main_logger，避免导入错误
# 初始时不会创建文件，只有在实际使用时才会配置
main_logger = logging.getLogger("HMASD")
