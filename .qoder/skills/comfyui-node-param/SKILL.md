---
name: comfyui-node-param
description: 为 ComfyUI 节点添加或修改参数的标准流程。适用于修改 qwen_3.py、qwen_25.py 等节点文件中的参数定义。
---

# ComfyUI 节点参数修改技巧

## 核心规律

修改 ComfyUI 节点参数时，必须同步修改三处：

1. **INPUT_TYPES** - 定义参数的 UI 输入框（类型、默认值、范围）
2. **方法签名** - 接收参数的方法参数列表
3. **方法实现** - 在代码中使用该参数

## 关键规则

### 规则 1：参数顺序必须一致

INPUT_TYPES 中 required 参数的顺序 **必须** 与方法签名中参数的顺序完全一致。

```python
# INPUT_TYPES
"required": {
    "model_path": ...,
    "dtype": ...,
    "keep_model_loaded": ...,
    # 顺序: model_path -> dtype -> keep_model_loaded
}

# 方法签名 - 顺序必须完全相同
def caption(self, model_path, dtype, keep_model_loaded, ...):
```

### 规则 2：FLOAT 类型默认值必须是浮点数

```python
# 正确
"repetition_penalty": ("FLOAT", {"default": 1.0, ...})

# 错误 - 会导致默认值显示为 1 而不是 1.0
"repetition_penalty": ("FLOAT", {"default": 1, ...})
```

### 规则 3：INT 类型建议添加类型注解

```python
# 推荐
def caption(self, ..., video_fps: int = 16, ...):

# 不推荐 - 可能导致类型推断问题
def caption(self, ..., video_fps = 16, ...):
```

### 规则 4：可选参数放最后

optional 参数可以放在方法签名末尾，使用默认值：

```python
def caption(self, model_path, dtype, keep_model_loaded, 
            instruction: str = None, video_fps: int = 16, image=None):
```

## 修改流程

### 步骤 1：读取当前代码

确认 INPUT_TYPES 和方法签名的当前状态：

```python
# INPUT_TYPES
"required": {
    "model_path": ...,
    "dtype": ...,
    "max_side": ("INT", {"default": 512, ...}),
    # ↑ 最后一个 required 参数
}

def caption(self, model_path, dtype, max_side=512, ...):
```

### 步骤 2：添加参数到 INPUT_TYPES

在 required 末尾添加新参数：

```python
"required": {
    "model_path": ...,
    "dtype": ...,
    "max_side": ("INT", {"default": 512, ...}),
    "max_new_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
}
```

### 步骤 3：同步修改方法签名

保持顺序一致：

```python
# 错误 - 顺序与 INPUT_TYPES 不一致
def caption(self, model_path, max_new_tokens, dtype, ...):

# 正确 - 顺序一致
def caption(self, model_path, dtype, max_new_tokens, ...):
```

### 步骤 4：在实现中使用参数

找到 `model.generate()` 调用并添加参数：

```python
generated_ids = self.model.generate(
    **inputs,
    max_new_tokens=max_new_tokens,
    repetition_penalty=repetition_penalty,
)
```

### 步骤 5：验证

运行 `get_problems` 检查语法错误：

```python
get_problems(["path/to/file.py"])
```

## 常见问题

### 问题：默认值显示错位

原因：方法签名参数顺序与 INPUT_TYPES 不一致。

解决：确保两处顺序完全相同。

### 问题：FLOAT 默认值显示为整数

原因：默认值写成了 `1` 而不是 `1.0`。

解决：使用浮点数格式 `1.0`。

### 问题：浏览器缓存导致显示旧值

解决：Ctrl+F5 强制刷新，或重启 ComfyUI。

## 常用参数类型

| 类型 | 默认值格式 | 示例 |
|------|-----------|------|
| INT | 整数 | `1024`, `16` |
| FLOAT | 浮点数 | `1.0`, `0.5` |
| BOOLEAN | True/False | `True`, `False` |
| STRING | 字符串 | `"default text"` |
