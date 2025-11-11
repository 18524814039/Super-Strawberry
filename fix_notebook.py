"""
修复notebook: 补充8.2 LSTM模型评估的代码
"""

import json

def fix_missing_lstm_eval():
    # 读取notebook
    with open('training_optimized.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']

    # 找到 "## 8.2 LSTM模型评估" 的位置
    insert_index = None
    for i, cell in enumerate(cells):
        if 'cell_type' in cell and cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if '## 8.2 LSTM模型评估' in source:
                insert_index = i + 1
                # 先把这个cell改为markdown
                cells[i] = {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": ["## 8.2 LSTM模型评估"]
                }
                print(f"✅ 找到位置，索引: {i}")
                break

    if insert_index is None:
        print("❌ 未找到 '## 8.2 LSTM模型评估'")
        return

    # 插入LSTM评估代码
    lstm_eval_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"\\n加载最佳模型...\")\n",
            "\n",
            "# 加载最佳模型\n",
            "trainer.load_checkpoint('best_model.pth')\n",
            "\n",
            "print(\"\\n\" + \"=\"*60)\n",
            "print(\"📊 评估 LSTM 模型\")\n",
            "print(\"=\"*60)\n",
            "\n",
            "# 完整评估\n",
            "metrics, predictions, targets, inputs = evaluate_model(\n",
            "    model=trainer.model,\n",
            "    data_loader=test_loader,\n",
            "    device=DEVICE,\n",
            "    save_dir=RESULTS_DIR\n",
            ")"
        ]
    }

    # 插入到指定位置
    cells.insert(insert_index, lstm_eval_code)
    print(f"✅ 插入LSTM评估代码在索引 {insert_index}")

    # 保存更新后的notebook
    with open('training_optimized.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)

    print("\n" + "="*60)
    print("✅ Notebook修复完成！")
    print("="*60)
    print("已添加:")
    print("- Cell 24: LSTM模型评估代码")
    print("  - 加载最佳模型")
    print("  - 运行evaluate_model()")
    print("  - 生成可视化结果")
    print("="*60)

if __name__ == '__main__':
    try:
        fix_missing_lstm_eval()
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
