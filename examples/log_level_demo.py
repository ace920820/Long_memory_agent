#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志级别控制演示
"""

import logging
import sys
import os

# 将项目根目录添加到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入日志配置工具
from utils import (
    configure_logging,
    set_log_level,
    set_development_logging,
    set_production_logging
)

def main():
    """日志级别控制演示主函数"""
    
    print("1. 默认情况下使用 INFO 级别")
    configure_logging(logging.INFO)
    
    # 获取根日志记录器
    logger = logging.getLogger()
    
    # 显示各级别日志
    logger.debug("这是一条 DEBUG 日志（不会显示）")
    logger.info("这是一条 INFO 日志（会显示）")
    logger.warning("这是一条 WARNING 日志（会显示）")
    
    # 切换到DEBUG级别
    print("\n2. 切换到 DEBUG 级别")
    set_log_level(logging.DEBUG)
    
    logger.debug("现在这条 DEBUG 日志会显示")
    logger.info("这是一条 INFO 日志（会显示）")
    
    # 使用便捷函数切换到生产环境级别（WARNING）
    print("\n3. 使用便捷函数切换到生产环境级别（WARNING）")
    set_production_logging()
    
    logger.debug("这条 DEBUG 日志不会显示")
    logger.info("这条 INFO 日志也不会显示")
    logger.warning("只有这条 WARNING 日志会显示")
    logger.error("ERROR 日志也会显示")
    
    # 使用便捷函数切换到开发环境级别（DEBUG）
    print("\n4. 使用便捷函数切换到开发环境级别（DEBUG）")
    set_development_logging()
    
    logger.debug("现在这条 DEBUG 日志会显示")
    
    # 使用字符串设置日志级别
    print("\n5. 使用字符串设置日志级别")
    set_log_level("INFO")
    
    logger.debug("这条 DEBUG 日志不会显示")
    logger.info("这条 INFO 日志会显示")
    
    print("\n6. 示例完成")

if __name__ == "__main__":
    main()
