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




# 附 参考文章

![SayHelloCode-vLLM系列](https://zhuanlan.zhihu.com/p/680153425)
