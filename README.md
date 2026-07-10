# Learning-DL

这是一个用于学习深度学习（Deep Learning）相关算法、模型与实践的仓库。仓库内包含多个领域的代码实现与讲义（PPTX 格式），涵盖计算机视觉（CV）、生成对抗网络（GAN）、自然语言处理（NLP）和强化学习（RL）等内容。

## 项目结构

本仓库的目录结构和主要学习模块如下：

*   **[full_connect.py](full_connect.py)**: 全连接神经网络的 TensorFlow 实现（以 notMNIST 数据集为例）。
*   **[cv/](cv)**: 计算机视觉（Computer Vision）
    *   [convnet.py](cv/convnet.py): 卷积神经网络的基础实现。
    *   [convnet_face.py](cv/convnet_face.py): 包含人脸检测/识别相关的 CNN 实践。
    *   [facesign/](cv/facesign): 使用微软 Azure 认知服务（Cognitive Face）的人脸识别与签到实现。
    *   [CNN.ipynb](cv/CNN.ipynb): 卷积神经网络的 Jupyter Notebook 练习。
    *   包含 ResNet, CapsNet, DeepFakes, StyleTransfer, ImageCaptioning 等经典方向 of 讲义。
*   **[gan/](gan)**: 生成对抗网络（Generative Adversarial Networks）
    *   [dcgan_mnist.py](gan/dcgan_mnist.py): 深度卷积生成对抗网络（DCGAN）在 MNIST 数据集上的实现。
    *   包含 GAN 基础概念的讲义。
*   **[nlp/](nlp)**: 自然语言处理（Natural Language Processing）
    *   [skipgram.py](nlp/skipgram.py) / [skipgram_cn.py](nlp/skipgram_cn.py): Word2Vec 中的 Skip-gram 模型实现（包括中文版）。
    *   [lstm.py](nlp/lstm.py) / [lstm_cn.py](nlp/lstm_cn.py) / [lstm_emb.py](nlp/lstm_emb.py): 循环神经网络（LSTM）文本生成与序列建模实现。
    *   [autotriage/](nlp/autotriage): 自动化分诊分类系统（基于 NLTK、Pandas 和 LSTM）。
    *   [BLEU.py](nlp/BLEU.py): 机器翻译评估指标 BLEU (Bilingual Evaluation Understudy) 的实现。
    *   包含 Word Embedding, Seq2Seq, Attention, Tacotron (语音合成), R-net 等前沿讲义。
*   **[rl/](rl)**: 强化学习（Reinforcement Learning）
    *   [frozenLakeQ.py](rl/frozenLakeQ.py): 经典的冰湖（Frozen Lake）Q-Learning 算法实现。
    *   [gobang/](rl/gobang): 五子棋强化学习实现与环境。
    *   [RL.ipynb](rl/RL.ipynb): 强化学习 Jupyter Notebook 练习。
*   **[tools/](tools)**: 工具与基础
    *   [Keras.ipynb](tools/Keras.ipynb): Keras 深度学习框架快速上手。
    *   [tSNE.py](tools/tSNE.py): t-SNE 数据降维与可视化实现。

---

## 依赖管理 (使用 `uv`)

为了提供极其快速、隔离且一致的运行环境，本项目全面采用了 Rust 编写的 Python 包管理利器 [uv](https://github.com/astral-sh/uv)。

### 1. 安装 `uv`

如果你尚未安装 `uv`，可以通过以下指令快速安装：

*   **Windows (PowerShell)**:
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
*   **macOS / Linux**:
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

### 2. 环境初始化

在项目根目录下，`uv` 会自动读取 `[.python-version](.python-version)`（固定为 `3.11` 版本以保证对大部分主流 DL 框架的最优兼容性）以及 `[pyproject.toml](pyproject.toml)` 进行环境同步：

```bash
# 同步依赖并自动创建/配置虚拟环境 (.venv)
uv sync
```

该命令会在当前目录下自动创建 `.venv` 虚拟环境，并把所有必需的依赖安装到位。

### 3. 运行代码

为了保证代码运行在隔离的虚拟环境中，使用 `uv run` 执行任意脚本：

```bash
# 示例：运行全连接神经网络脚本
uv run python full_connect.py
```

### 4. 运行 Jupyter Notebook

如果你需要运行项目中的 `.ipynb` 文件：

```bash
# 安装 Jupyter 工具（如果尚未添加）
uv add jupyter

# 启动 Jupyter Notebook
uv run jupyter notebook
```

### 5. 添加新依赖

如需在此学习项目中引入新的第三方库：

```bash
uv add <package_name>
```
这会自动更新 `[pyproject.toml](pyproject.toml)` 并锁定版本到 `[uv.lock](uv.lock)` 文件中。
