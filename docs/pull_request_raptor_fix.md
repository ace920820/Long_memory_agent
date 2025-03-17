# 修复Raptor搜索模块中的嵌入模型访问问题

## 问题描述

在使用RaptorModule的search方法时，代码尝试访问`self.RA.embedding_model.get_embedding`方法，但在RetrievalAugmentation类中并没有直接定义`embedding_model`属性，导致代码无法正常运行。

## 解决方案

1. 修改了嵌入向量获取方式，使用`self.RA.retriever.create_embedding`方法替代不存在的`self.RA.embedding_model.get_embedding`
2. 增强了`distances_from_embeddings`函数以处理可能为None的嵌入向量
3. 添加了详细的错误处理和日志输出，提高系统健壮性

## 主要更改

- `models/raptor_module.py`: 修复嵌入向量获取方法，添加错误处理
- `raptor/utils.py`: 增强距离计算函数，支持处理None值
- 添加测试脚本和文档

## 测试情况

- 创建了`test_direct_raptor.py`脚本测试修改后的距离计算功能
- 验证了当嵌入向量为None时的处理逻辑
- 测试了正常情况下的距离计算和重排序功能

## 注意事项

- 该修复依赖于TreeRetriever中的嵌入模型，如果后续修改TreeRetriever的实现，需要相应地更新RaptorModule

## 相关文档

详细的修复文档请参考 [docs/raptor_search_fix.md](./raptor_search_fix.md)
