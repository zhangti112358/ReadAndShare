# 内存相关

## PagedAttention

单次推理分配的kv-cache内存：prompt+output(按最大输出tokens数量分配)
其中存在内存浪费的情况有：
1. 内部碎片：给输出分配，但没有被使用的部分。
2. 外部碎片：内存块不足一个推理分配值，无法使用的情况。
3. 多个推理序列：可能存在相同的tokens，重复占用空间。

解决方法：
1. tokens分块存储。比如16个tokens一块，单块最多浪费16-1个空间。
2. 分块之间建立表格。逻辑连续的块，物理地址无需连续。
3. 序列中前面部分相同，从某个位置不同的情况。可以共享前面一部分块。比如parallel sampling，beam search，相同prompt情况。

## FlashAttention

加速Attention计算 Attention(Q,K,V) = softmax(QK'/sqrt(d))V

1. 单独计算每一步，需要读写多次中间结果到显存。为了加速，使用计算融合，中间结果存在共享内存（SRAM）上。
2. 共享内存无法存下整个矩阵的中间数据，所以计算要分块。
3. 计算分块，而softmax计算需要向量的所有值，产生矛盾。解决方法是先计算单块的softmax，并记录此块的求和，当所有分块的softmax计算完成后，更新之前分块的softmax值。单块的softmax和全局softmax值差异为除的分母不同，所以乘一个系数即可得到最终结果。

# 模型结构

## MOE 架构

将FFN计算分为多个专家，根据数据特征选择部分专家计算，减少计算量。
1. 采用路由机制选择专家。
2. 分共享专家（所有数据都计算的）和路由专家（选择计算）。
3. 多个专家可能存在负载不平衡问题。在选择模块加入可学习参数，平衡多个专家的被选择概率

## 多头隐式注意力 (Multi-Head Latent Attention)

目标：减少KV cache存储。比如：
1. GQA，MQA：一个KV对应多个Q，降低KV数量。
2. MLA：对KV矩阵做低秩分解，得到低维隐向量。计算时升维得到原始数据。

## 多令牌预测

一次性生成多个tokens。

1. 主模型计算第一个token。
2. MTP模型计算第2个token。（和主模型共用embedding层和输出层，中间计算量更小）
3. MTP模型2计算第3个token，以此类推。

DeepSeek V3：训练时使用MTP可以提升回复质量，推理时未使用。

## 计算精度 混合精度训练

# 附 参考文章

![SayHelloCode-vLLM系列](https://zhuanlan.zhihu.com/p/680153425)

![Attention加速](https://zhuanlan.zhihu.com/p/638468472)

![DeepSeek关键技术详解](https://zhuanlan.zhihu.com/p/23048347789)