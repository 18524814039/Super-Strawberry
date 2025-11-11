"""
更新training_optimized.ipynb以使用修复后的代码
主要改动：
1. INPUT_DIM从5改为3（只使用signal_0, signal_1, signal_2）
2. 添加git pull步骤
3. 添加baseline评估
4. 更新对比分析
"""

import json
import sys

def update_notebook():
    # 读取notebook
    with open('training_optimized.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # 找到并更新Cell 2 (克隆仓库) - 添加git pull
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code' and 'Super-Strawberry' in str(cell.get('source', '')):
            if 'git clone' in str(cell['source']):
                # 更新为先pull后clone
                cell['source'] = [
                    "# 如果在 Colab 中运行，先克隆仓库\n",
                    "import os\n",
                    "\n",
                    "if not os.path.exists('Super-Strawberry'):\n",
                    "    # 克隆仓库\n",
                    "    !git clone https://github.com/18524814039/Super-Strawberry.git\n",
                    "    print(\"✅ 仓库克隆成功\")\n",
                    "else:\n",
                    "    print(\"✅ 仓库已存在，拉取最新代码...\")\n",
                    "    !cd Super-Strawberry && git pull origin claude/robotic-arm-torque-prediction-011CUr36WDgDiXoWBgS6LoVt\n",
                    "\n",
                    "# 切换到项目目录\n",
                    "%cd Super-Strawberry"
                ]
                print(f"✅ 更新Cell {i}: 添加git pull")
                break

    # 找到并更新Cell 5 (导入模块) - 添加evaluate_baseline
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code' and 'from src.evaluate import' in str(cell.get('source', '')):
            cell['source'] = [
                "import sys\n",
                "import os\n",
                "import torch\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# 添加项目根目录到路径\n",
                "project_root = os.path.abspath('.')\n",
                "if project_root not in sys.path:\n",
                "    sys.path.insert(0, project_root)\n",
                "\n",
                "# 导入自定义模块\n",
                "from src.data_loader import load_data_dynamic, get_sample_data_info\n",
                "from src.model import get_model\n",
                "from src.train import Trainer\n",
                "from src.evaluate import evaluate_model, evaluate_baseline, plot_training_history\n",
                "\n",
                "print(f\"✅ PyTorch version: {torch.__version__}\")\n",
                "print(f\"✅ CUDA available: {torch.cuda.is_available()}\")\n",
                "if torch.cuda.is_available():\n",
                "    print(f\"✅ CUDA device: {torch.cuda.get_device_name(0)}\")\n",
                "\n",
                "# 设置随机种子\n",
                "torch.manual_seed(42)\n",
                "np.random.seed(42)"
            ]
            print(f"✅ 更新Cell {i}: 添加evaluate_baseline导入")
            break

    # 找到并更新Cell 7 (配置参数) - 改INPUT_DIM为3
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code' and 'INPUT_DIM = 5' in str(cell.get('source', '')):
            # 替换INPUT_DIM
            source_str = ''.join(cell['source'])
            source_str = source_str.replace('INPUT_DIM = 5', 'INPUT_DIM = 3')
            source_str = source_str.replace(
                '输入特征维度（Time, Torque, signal_0, signal_1, signal_2）',
                '输入特征维度（signal_0, signal_1, signal_2）'
            )
            source_str = source_str.replace(
                'INPUT_DIM} (Time, Torque, signal_0, signal_1, signal_2)',
                'INPUT_DIM} (仅signal_0, signal_1, signal_2)'
            )
            cell['source'] = source_str.split('\n')
            # 保持每行的换行符
            cell['source'] = [line + '\n' if i < len(cell['source'])-1 else line
                             for i, line in enumerate(cell['source'])]
            print(f"✅ 更新Cell {i}: INPUT_DIM改为3")
            break

    # 在Cell 21评估之前插入baseline评估
    insert_index = None
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'markdown' and '## 8. 模型评估' in str(cell.get('source', '')):
            insert_index = i + 1
            break

    if insert_index:
        # 插入baseline评估标题
        baseline_markdown = {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 8.1 Baseline评估（用于对比）\n", "\n", "先评估最简单的Persistence Baseline，用于判断LSTM是否真正学到了东西。"]
        }

        # 插入baseline评估代码
        baseline_code = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"📊 评估 Persistence Baseline\")\n",
                "print(\"=\"*60)\n",
                "print(\"策略: 用输入序列最后一个signal_1值预测所有输出\")\n",
                "print(\"如果LSTM连这个都打不过，说明模型完全失败\")\n",
                "print(\"=\"*60)\n",
                "\n",
                "baseline_metrics = evaluate_baseline(test_loader, device=DEVICE)\n",
                "\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"📊 Baseline评估完成\")\n",
                "print(\"=\"*60)\n",
                "print(f\"Baseline R²:   {baseline_metrics['R2']:.6f}\")\n",
                "print(f\"Baseline MAPE: {baseline_metrics['MAPE']:.2f}%\")\n",
                "print(\"=\"*60)"
            ]
        }

        # 更新标题
        cells[insert_index]['source'] = ["## 8.2 LSTM模型评估"]

        # 插入新cells
        cells.insert(insert_index, baseline_markdown)
        cells.insert(insert_index + 1, baseline_code)
        print(f"✅ 插入baseline评估在索引 {insert_index}")

    # 更新Cell 23 (结果对比) - 使用真实baseline而不是硬编码
    for i, cell in enumerate(cells):
        if cell.get('cell_type') == 'code' and 'baseline_r2 = 0.949' in str(cell.get('source', '')):
            cell['source'] = [
                "# 打印详细指标\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"📊 LSTM 最终评估指标\")\n",
                "print(\"=\"*60)\n",
                "\n",
                "for key, value in metrics.items():\n",
                "    if key == 'R2':\n",
                "        status = '✅ 优秀' if value > 0.5 else '⚠️ 一般' if value > 0.3 else '❌ 较差'\n",
                "        print(f\"{key:10s}: {value:.6f}  ({status})\")\n",
                "    elif key == 'MAPE':\n",
                "        status = '✅ 优秀' if value < 30 else '⚠️ 一般' if value < 50 else '❌ 较差'\n",
                "        print(f\"{key:10s}: {value:.2f}%    ({status})\")\n",
                "    else:\n",
                "        print(f\"{key:10s}: {value:.6f}\")\n",
                "\n",
                "print(\"=\"*60)\n",
                "\n",
                "# 与真实baseline对比\n",
                "print(\"\\n\" + \"=\"*60)\n",
                "print(\"📊 LSTM vs Baseline 对比\")\n",
                "print(\"=\"*60)\n",
                "print(f\"{'指标':<10} {'LSTM':<15} {'Baseline':<15} {'LSTM改进':<15}\")\n",
                "print(\"-\"*60)\n",
                "\n",
                "for key in ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE']:\n",
                "    lstm_val = metrics[key]\n",
                "    baseline_val = baseline_metrics[key]\n",
                "    \n",
                "    if key in ['MSE', 'RMSE', 'MAE', 'MAPE']:\n",
                "        # 越小越好\n",
                "        improvement = (baseline_val - lstm_val) / baseline_val * 100\n",
                "        symbol = \"✅\" if improvement > 0 else \"❌\"\n",
                "    else:  # R2\n",
                "        # 越大越好\n",
                "        if baseline_val != 0:\n",
                "            improvement = (lstm_val - baseline_val) / abs(baseline_val) * 100\n",
                "        else:\n",
                "            improvement = 0\n",
                "        symbol = \"✅\" if improvement > 0 else \"❌\"\n",
                "    \n",
                "    print(f\"{key:<10} {lstm_val:<15.4f} {baseline_val:<15.4f} {improvement:>+7.1f}% {symbol}\")\n",
                "\n",
                "print(\"=\"*60)\n",
                "\n",
                "# 总结\n",
                "print(\"\\n💡 总结:\")\n",
                "if metrics['R2'] > baseline_metrics['R2']:\n",
                "    print(\"✅ LSTM表现优于Baseline，模型有效！\")\n",
                "elif metrics['R2'] > baseline_metrics['R2'] * 0.9:\n",
                "    print(\"⚠️ LSTM略优于Baseline，还有改进空间\")\n",
                "else:\n",
                "    print(\"❌ LSTM不如Baseline，模型设计可能有问题\")"
            ]
            print(f"✅ 更新Cell {i}: 使用真实baseline对比")
            break

    # 保存更新后的notebook
    with open('training_optimized.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("✅ Notebook更新完成！")
    print("="*60)
    print("主要改动:")
    print("1. INPUT_DIM: 5 → 3 (只使用signal_0/1/2)")
    print("2. 添加git pull步骤（自动更新代码）")
    print("3. 添加Baseline评估")
    print("4. 使用真实baseline进行对比（而不是硬编码的0.949）")
    print("="*60)

if __name__ == '__main__':
    try:
        update_notebook()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
