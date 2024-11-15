# 用于股票交易的实用深度强化学习方法

## 请查看 FinRL 库
现在，该项目已合并到 [FinRL 库](https://github.com/AI4Finance-LLC/FinRL-Library) 中

## 先决条件
Python 3.6+ 环境

## 第 1 步：安装 OpenAI Baselines 系统包 [OpenAI 说明](https://github.com/openai/baselines)
### Ubuntu
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```
### Mac OS X
在 Mac 上安装系统包需要使用 [Homebrew](https://brew.sh)。安装好 Homebrew 后，运行以下命令：
```bash
brew install cmake openmpi
```


## 第 2 步：创建和激活虚拟环境
将仓库克隆到文件夹 /DQN-DDPG_Stock_Trading：
```bash
git clone https://github.com/hust512/DQN-DDPG_Stock_Trading.git
cd DQN-DDPG_Stock_Trading
```
在文件夹 /DQN-DDPG_Stock_Trading 下，创建一个虚拟环境
```bash
pip install virtualenv
```
虚拟环境本质上是包含 Python 可执行文件和所有 Python 包副本的文件夹。
在文件夹 /DQN-DDPG_Stock_Trading/venv 下创建一个名为 venv 的虚拟环境
```bash
virtualenv -p python3 venv
```
要激活虚拟环境：
```
source venv/bin/activate
```

## 第 3 步：在此虚拟环境下安装 openAI gym 环境：venv
#### TensorFlow 版本
主分支支持 TensorFlow 1.4 到 1.14 版本。要支持 TensorFlow 2.0，请使用 tf2 分支。有关更多详细信息，请参阅 [TensorFlow 安装指南](https://www.tensorflow.org/install/)。
- 安装 gym 和 TensorFlow 包：
    ```bash
    pip install gym
    pip install gym[atari] 
    pip install tensorflow==1.14
    ```
- 可能缺少的其他包：
    ```bash
    pip install filelock
    pip install matplotlib
    pip install pandas
    ```

## 第 4 步：下载并安装官方基线包
- 将基线仓库克隆到文件夹 DQN-DDPG_Stock_Trading/baselines：
    ```bash
    git clone https://github.com/openai/baselines.git
    cd baselines
    ```

- 安装基线包
    ```bash
    pip install -e .
    ```

## 第 5 步：测试安装
运行基线中的所有单元测试：
```
pip install pytest
pytest
```
预期结果类似于 '94 passed, 49 skipped, 72 warnings in 355.29s'。如果需要修复失败的测试，请查看 OpenAI 基线的 [问题](https://github.com/openai/baselines/issues) 或 Stack Overflow。

## 第 6 步：测试 OpenAI Atari Pong 游戏
### 如果这可以正常工作，那么就可以开始实现股票交易应用程序了
```bash
python -m baselines.run --alg=ppo2 --env=PongNoFrameskip-v4 --num_timesteps=1e4 --load_path=~/models/pong_20M_ppo2 --play
```
预期每集的平均奖励约为 20。

## 第 7 步：在 gym 中注册股票交易环境

在文件夹 /DQN-DDPG_Stock_Trading/venv 中注册 RLStock-v0 环境：
从
```bash
DQN-DDPG_Stock_Trading/gym/envs/__init__.py
```
复制以下内容：
```bash
register(
    id='RLStock-v0',
    entry_point='gym.envs.rlstock:StockEnv',
)
register(
    id='RLTestStock-v0',
    entry_point='gym.envs.rlstock:StockTestEnv',
)
```
到 venv gym 环境：
```bash
/DQN-DDPG_Stock_Trading/venv/lib/python3.6/site-packages/gym/envs/__init__.py
```

## 第 8 步：在 gym 中构建股票交易环境

- 复制文件夹
```bash
DQN_Stock_Trading/gym/envs/rlstock
```
到 venv gym 环境文件夹：
```bash
/DQN-DDPG_Stock_Trading/venv/lib/python3.6/site-packages/gym/envs
```

- 打开
```bash
/DQN-DDPG_Stock_Trading/venv/lib/python3.6/site-packages/gym/envs/rlstock/rlstock_env.py 
/DQN-DDPG_Stock_Trading/venv/lib/python3.6/site-packages/gym/envs/rlstock/rlstock_testenv.py
```
更改这两个文件中的导入数据路径（进入 rlstock 文件夹并使用 pwd 检查文件夹路径）。
### 基线
替换
```bash
/DQN-DDPG_Stock_Trading/baselines/baselines/run.py
```
为
```bash
/DQN-DDPG_Stock_Trading/run.py
```

## 第 9 步：训练模型和测试

### 预先步骤：
转到文件夹
```
/DQN-DDPG_Stock_Trading/
```
激活虚拟环境
```
source venv/bin/activate
```
转到基线文件夹
```
/DQN-DDPG_Stock_Trading/baselines
```
### 训练
要训练模型，运行此命令
```bash
python -m baselines.run --alg=ddpg --env=RLStock-v0 --network=mlp --num_timesteps=1e4
```
### 交易
要查看测试/交易结果，请运行此命令
```bash
python -m baselines.run --alg=ddpg --env=RLStock-v0 --network=mlp --num_timesteps=2e4 --play
```

结果图像位于文件夹 /DQN-DDPG_Stock_Trading/baselines 下。

（可以调整超参数 num_timesteps 以更好地训练模型，注意如果此数字过高，则会导致过拟合问题；如果过低，则会导致欠拟合问题。）

与我们的结果比较：

<img src=result_trading.png width="500">


### 可能需要的其他命令：
```bash
pip3 install opencv-python
pip3 install lockfile
pip3 install -U numpy
pip3 install mujoco-py==0.5.7
```

#### 请引用以下论文
Xiong, Z., Liu, X.Y., Zhong, S., Yang, H. 和 Walid, A.，2018 年。用于股票交易的实用深度强化学习方法，NeurIPS 2018 AI in Finance 研讨会。
