# Raptor模块查询功能修复文档

## 问题描述

在RaptorModule的search方法中，尝试访问`RA.embedding_model.get_embedding`方法，但在RetrievalAugmentation类中并没有直接定义`embedding_model`属性，导致无法正常获取嵌入向量进行相似度计算和重排序。

## 解决方案

通过分析发现，嵌入模型实际上是在TreeRetriever对象中定义的。因此，我们修改了代码，使用`RA.retriever.create_embedding`方法替代不存在的`RA.embedding_model.get_embedding`方法。

## 关键修改

### 1. RaptorModule.search 方法

在RaptorModule的search方法中，修改了获取嵌入向量的方式：

```python
# 修改前
query_embedding = self.RA.embedding_model.get_embedding(query)
node_embeddings = [self.RA.embedding_model.get_embedding(text) for text in node_texts]

# 修改后
try:
    query_embedding = self.RA.retriever.create_embedding(query)
    
    node_embeddings = []
    for text in node_texts:
        try:
            node_embeddings.append(self.RA.retriever.create_embedding(text))
        except Exception as e:
            logging.error(f"获取文本嵌入向量时发生错误: {str(e)}")
            node_embeddings.append(None)
except Exception as e:
    logging.error(f"获取查询嵌入向量时发生错误: {str(e)}")
    # 继续处理错误情况...
```

### 2. distances_from_embeddings 函数增强

增强了`distances_from_embeddings`函数以处理可能为None的嵌入向量：

```python
# 检查嵌入向量是否有效
if query_embedding is None:
    logging.error("查询嵌入向量为None，无法计算距离")
    return [1.0] * len(embeddings)  # 返回最大距离
    
# 过滤掉None值的嵌入向量，同时记录对应的索引
valid_embeddings = []
valid_indices = []
for i, emb in enumerate(embeddings):
    if emb is not None:
        valid_embeddings.append(emb)
        valid_indices.append(i)
        
if not valid_embeddings:
    logging.error("没有有效的嵌入向量，无法计算距离")
    return [1.0] * len(embeddings)  # 返回最大距离
```

### 3. 错误处理和日志输出

在所有关键操作点增加了错误处理和日志输出，便于排查问题：

```python
try:
    # 执行操作...
except Exception as e:
    logging.error(f"操作描述时发生错误: {str(e)}")
    # 错误恢复逻辑...
```

## 测试验证

创建了两个测试脚本：

1. `test_raptor_search.py`：全面测试RaptorModule的search方法
2. `test_direct_raptor.py`：直接测试修改后的distances_from_embeddings函数

测试结果表明：
- 当嵌入向量为None时，距离计算函数可以返回默认的最大距离值1.0
- 正常情况下距离计算可以正确处理有效的嵌入向量
- 即使部分嵌入向量为None，函数也能够正常处理剩余的有效向量

## 总结

通过修改嵌入向量获取方式、增强距离计算函数和添加错误处理，解决了RaptorModule搜索功能中的嵌入模型访问问题。修复后的代码可以更健壮地处理各种异常情况，提升了系统的稳定性。

## 后续建议

1. 考虑在RetrievalAugmentation类初始化时，提供一个统一的接口访问嵌入模型，避免需要通过retriever间接访问
2. 添加更多单元测试，确保各种边缘情况都能被正确处理
3. 优化重排序过程，考虑预加载模型以提高性能

# Raptor 搜索和重排序功能修复文档

## 问题描述

在 Raptor 系统中，RaptorModule 的 `search` 方法调用了不存在的 `RA.embedding_model.get_embedding` 方法来获取嵌入向量，导致在执行搜索时出错。此外，重排序功能在处理嵌入为 `None` 的情况时也存在问题。

## 修复内容

### 1. RaptorModule 类的 search 方法
- 将 `RA.embedding_model.get_embedding` 修改为 `RA.retriever.create_embedding`
- 增强日志记录，添加详细的调试信息
- 改进错误处理，确保在获取嵌入向量失败时能够优雅地处理
- 增加了对查询执行过程的完整跟踪日志

### 2. distances_from_embeddings 函数
- 添加对 `None` 值嵌入向量的处理
- 改进日志记录，包括处理的嵌入向量数量和有效嵌入的比例
- 增加对重排序前后排名变化的跟踪

### 3. RaptorConfigManager 类
- 增加 `is_rerank_enabled` 方法，简化对重排序功能是否启用的判断

## 测试结果

创建了两个测试脚本来验证修复：

1. `test_direct_raptor.py` - 测试 `distances_from_embeddings` 函数的基本功能和处理 `None` 值的能力
2. `test_raptor_module.py` - 测试修复后的 RaptorModule 类的搜索和重排序功能

测试结果表明：
- 修复后的代码能够正确获取嵌入向量
- 能够处理 `None` 值嵌入向量
- 重排序功能可以正常工作
- 日志记录更加详细，有助于排查问题

## 代码变更

### RaptorModule 修改
```python
# 旧实现
query_embedding = self.RA.embedding_model.get_embedding(query)

# 新实现
query_embedding = self.RA.retriever.create_embedding(query)
```

### distances_from_embeddings 修改
```python
# 增加对 None 值的处理
valid_embeddings = [(i, e) for i, e in enumerate(embeddings) if e is not None]
logging.info(f"处理嵌入向量：总计 {len(embeddings)} 个，有效 {len(valid_embeddings)} 个")

if not valid_embeddings:
    logging.warning("没有有效的嵌入向量，返回全部最大距离")
    return [999.0] * len(embeddings)
```

## 后续工作

1. 进一步优化重排序功能的性能，特别是在处理大量文本时
2. 考虑增加对不同嵌入模型的适配
3. 完善配置管理，使模型加载更加灵活
