#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志级别控制脚本
用法：
- 设置为调试级别：python set_log_level.py debug
- 设置为信息级别：python set_log_level.py info
- 设置为警告级别：python set_log_level.py warning
- 设置为错误级别：python set_log_level.py error
- 设置为开发环境：python set_log_level.py dev
- 设置为生产环境：python set_log_level.py prod
"""

import sys
import os
import logging

# 导入日志配置工具
from utils import (
    configure_logging,
    set_log_level,
    set_development_logging,
    set_production_logging
)

def print_help():
    """打印帮助信息"""
    print("日志级别控制脚本")
    print("用法：")
    print("  python set_log_level.py <日志级别>")
    print("可用的日志级别：")
    print("  debug   - 调试级别（非常详细）")
    print("  info    - 信息级别（默认）")
    print("  warning - 警告级别")
    print("  error   - 错误级别")
    print("  dev     - 开发环境（相当于 debug）")
    print("  prod    - 生产环境（相当于 warning）")
    print("\n示例：")
    print("  python set_log_level.py debug")
    print("  python set_log_level.py prod")
    print("\n注意：此脚本只影响环境变量，需要重启应用程序生效")

def main():
    """主函数"""
    if len(sys.argv) != 2:
        print_help()
        return
    
    level = sys.argv[1].lower()
    
    if level == "debug":
        # 设置环境变量
        os.environ["LOG_LEVEL"] = "DEBUG"
        print("已设置日志级别为 DEBUG（调试级别）")
        print("此设置将在下次启动应用程序时生效")
    elif level == "info":
        os.environ["LOG_LEVEL"] = "INFO"
        print("已设置日志级别为 INFO（信息级别）")
        print("此设置将在下次启动应用程序时生效")
    elif level == "warning":
        os.environ["LOG_LEVEL"] = "WARNING"
        print("已设置日志级别为 WARNING（警告级别）")
        print("此设置将在下次启动应用程序时生效")
    elif level == "error":
        os.environ["LOG_LEVEL"] = "ERROR"
        print("已设置日志级别为 ERROR（错误级别）")
        print("此设置将在下次启动应用程序时生效")
    elif level == "dev":
        os.environ["LOG_LEVEL"] = "DEBUG"
        print("已设置为开发环境（DEBUG级别）")
        print("此设置将在下次启动应用程序时生效")
    elif level == "prod":
        os.environ["LOG_LEVEL"] = "WARNING"
        print("已设置为生产环境（WARNING级别）")
        print("此设置将在下次启动应用程序时生效")
    else:
        print(f"错误：未知的日志级别 '{level}'")
        print_help()

if __name__ == "__main__":
    main()
