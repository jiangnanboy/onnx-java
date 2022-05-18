这里利用java加载onnx模型，并进行推理。

#### 步骤

1.利用java加载onnx模型，并进行推理预测。这里采用roberta模型的onnx版。

2.pytorch模型从[这里](https://huggingface.co/uer/chinese_roberta_L-2_H-512) 下载。

3.pytorch模型转为onnx见[这里](https://github.com/jiangnanboy/model2onnx) 。

#### 使用
1.sy/BertMask

```
String text = "中国的首都是[MASK]京。";
Triple<BertTokenizer, Map<String, OnnxTensor>, Integer> triple = null;
try {
    triple = parseInputText(text);
} catch (Exception e) {
    e.printStackTrace();
}
var maskPredictions = predMask(triple);
System.out.println(maskPredictions);
```

2.result
```
String text = "中国的首都是[MASK]京。";

tokens -> [[CLS], 中, 国, 的, 首, 都, 是, [MASK], 京, 。, [SEP]]
[MASK] predictions -> [北, 南, 东, 燕, 望]

String text = "我家后面有一[MASK]大树。";

tokens -> [[CLS], 我, 家, 后, 面, 有, 一, [MASK], 大, 树, 。, [SEP]]
[MASK] predictions -> [棵, 个, 株, 只, 颗]
```

#### 参考
https://github.com/jiangnanboy/model2onnx

https://huggingface.co/uer/chinese_roberta_L-2_H-512

https://arxiv.org/pdf/1907.11692.pdf

https://github.com/ankiteciitkgp/bertTokenizer

https://arxiv.org/pdf/1810.04805.pdf
