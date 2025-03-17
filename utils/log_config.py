#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志配置工具，提供统一的日志级别控制
"""

import logging
import sys

def configure_logging(level=logging.INFO, format_str=None):
    """
    配置全局日志级别和格式
    
    参数:
        level: 日志级别，可以是 logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
              或者字符串形式："DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
        format_str: 自定义日志格式，默认为 "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    """
    # 如果传入的是字符串，转换为对应的日志级别
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # 默认日志格式
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 配置根日志
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[
            logging.StreamHandler(sys.stdout)  # 输出到标准输出
        ],
        force=True  # 强制重新配置
    )
    
    # 降低一些常见的噪声较大的库的日志级别
    if level <= logging.DEBUG:
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('umap').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('transformers').setLevel(logging.WARNING)
        logging.getLogger('torch').setLevel(logging.WARNING)
    
    logging.info(f"日志级别已设置为: {logging.getLevelName(level)}")
    
    return logging.getLogger()

def get_logger(name, level=None):
    """
    获取一个配置好的日志记录器
    
    参数:
        name: 日志记录器名称
        level: 此记录器的日志级别，如果为None则使用根日志级别
    
    返回:
        Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    
    if level is not None:
        if isinstance(level, str):
            level = getattr(logging, level.upper())
        logger.setLevel(level)
    
    return logger

def set_log_level(level):
    """
    动态设置全局日志级别
    
    参数:
        level: 日志级别，可以是 logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL
              或者字符串形式："DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # 设置根日志级别
    logging.getLogger().setLevel(level)
    
    # 重新设置一些常见库的日志级别
    if level <= logging.DEBUG:
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('umap').setLevel(logging.WARNING)
    else:
        # 如果全局级别高于DEBUG，这些库也使用相同级别
        logging.getLogger('numba').setLevel(level)
        logging.getLogger('umap').setLevel(level)
    
    logging.info(f"全局日志级别已更改为: {logging.getLevelName(level)}")

# 便捷函数，设置为开发环境（详细日志）
def set_development_logging():
    """设置为开发环境（DEBUG级别的详细日志）"""
    configure_logging(logging.DEBUG)

# 便捷函数，设置为生产环境（较少的日志）
def set_production_logging():
    """设置为生产环境（WARNING级别的精简日志）"""
    configure_logging(logging.WARNING)

# 如果直接运行此脚本，则展示各级别的日志示例
if __name__ == "__main__":
    # 配置日志为DEBUG级别
    logger = configure_logging(logging.DEBUG)
    
    # 展示各级别的日志示例
    logger.debug("这是一条DEBUG级别的日志消息 - 非常详细的调试信息")
    logger.info("这是一条INFO级别的日志消息 - 一般信息")
    logger.warning("这是一条WARNING级别的日志消息 - 警告")
    logger.error("这是一条ERROR级别的日志消息 - 错误")
    logger.critical("这是一条CRITICAL级别的日志消息 - 严重错误")
    
    # 展示动态更改日志级别
    print("\n将日志级别更改为INFO")
    set_log_level("INFO")
    
    logger.debug("这条DEBUG消息不会显示")
    logger.info("这条INFO消息会显示")
    
    print("\n将日志级别更改为WARNING")
    set_log_level(logging.WARNING)
    
    logger.debug("这条DEBUG消息不会显示")
    logger.info("这条INFO消息也不会显示")
    logger.warning("这条WARNING消息会显示")
