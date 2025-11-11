"""
数据质量分析脚本
分析CSV数据中的重复行和无用特征
"""

import pandas as pd
import numpy as np
import glob
import os

def analyze_csv_quality(csv_file):
    """分析单个CSV文件的数据质量"""
    df = pd.read_csv(csv_file)

    # 跳过第一行（干扰数据）
    df = df.iloc[1:].reset_index(drop=True)

    results = {
        'filename': os.path.basename(csv_file),
        'total_rows': len(df),
    }

    # 1. 检查Time(s)列的唯一值
    time_unique = df['Time(s)'].nunique()
    time_values = df['Time(s)'].unique()
    results['time_unique_values'] = time_unique
    results['time_is_constant'] = (time_unique == 1)
    if time_unique <= 5:
        results['time_sample_values'] = time_values.tolist()

    # 2. 检查每列的唯一值比例
    for col in ['Time(s)', 'Torque', 'signal_0', 'signal_1', 'signal_2']:
        unique_count = df[col].nunique()
        unique_ratio = unique_count / len(df) * 100
        results[f'{col}_unique_ratio'] = unique_ratio

    # 3. 检查连续重复行
    # 对于signal_0, signal_1, signal_2，检查连续相同值的最长序列
    for col in ['signal_0', 'signal_1', 'signal_2']:
        values = df[col].values
        max_consecutive = 1
        current_consecutive = 1

        for i in range(1, len(values)):
            if values[i] == values[i-1]:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 1

        results[f'{col}_max_consecutive_duplicates'] = max_consecutive
        results[f'{col}_consecutive_dup_ratio'] = max_consecutive / len(df) * 100

    # 4. 检查完全重复的行（只看signal列）
    signal_df = df[['signal_0', 'signal_1', 'signal_2']]
    duplicate_rows = signal_df.duplicated().sum()
    results['duplicate_signal_rows'] = duplicate_rows
    results['duplicate_signal_ratio'] = duplicate_rows / len(df) * 100

    return results, df


def main():
    data_dir = './data'
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))

    if len(csv_files) == 0:
        print(f"❌ 未找到CSV文件在 {data_dir}")
        return

    print("=" * 80)
    print("🔍 数据质量分析")
    print("=" * 80)
    print(f"分析 {len(csv_files)} 个CSV文件...\n")

    # 分析前5个文件的详细信息
    print("📊 详细分析（前5个文件）:")
    print("=" * 80)

    all_results = []
    for i, csv_file in enumerate(csv_files[:5]):
        results, df = analyze_csv_quality(csv_file)
        all_results.append(results)

        print(f"\n文件 {i+1}: {results['filename']}")
        print(f"  总行数: {results['total_rows']}")
        print(f"\n  Time(s) 列:")
        print(f"    唯一值数量: {results['time_unique_values']}")
        print(f"    是否恒定: {'是 ⚠️' if results['time_is_constant'] else '否'}")
        if 'time_sample_values' in results:
            print(f"    值: {results['time_sample_values']}")

        print(f"\n  各列唯一值比例:")
        for col in ['Time(s)', 'Torque', 'signal_0', 'signal_1', 'signal_2']:
            ratio = results[f'{col}_unique_ratio']
            status = "✅" if ratio > 50 else "⚠️" if ratio > 10 else "❌"
            print(f"    {col:12s}: {ratio:6.2f}% {status}")

        print(f"\n  连续重复情况:")
        for col in ['signal_0', 'signal_1', 'signal_2']:
            max_dup = results[f'{col}_max_consecutive_duplicates']
            ratio = results[f'{col}_consecutive_dup_ratio']
            status = "✅" if ratio < 10 else "⚠️" if ratio < 30 else "❌"
            print(f"    {col:12s}: 最长{max_dup:4d}行连续重复 ({ratio:5.1f}%) {status}")

        print(f"\n  重复行统计:")
        dup_count = results['duplicate_signal_rows']
        dup_ratio = results['duplicate_signal_ratio']
        status = "✅" if dup_ratio < 10 else "⚠️" if dup_ratio < 30 else "❌"
        print(f"    signal列重复行: {dup_count}/{results['total_rows']} ({dup_ratio:.1f}%) {status}")

    # 统计所有文件
    print("\n\n" + "=" * 80)
    print("📈 整体统计（所有文件）:")
    print("=" * 80)

    all_results_full = []
    for csv_file in csv_files:
        results, _ = analyze_csv_quality(csv_file)
        all_results_full.append(results)

    # Time列恒定的文件比例
    time_constant_count = sum(1 for r in all_results_full if r['time_is_constant'])
    print(f"\nTime(s)列恒定的文件: {time_constant_count}/{len(csv_files)} ({time_constant_count/len(csv_files)*100:.1f}%)")

    # 平均唯一值比例
    print(f"\n各列平均唯一值比例:")
    for col in ['Time(s)', 'Torque', 'signal_0', 'signal_1', 'signal_2']:
        avg_ratio = np.mean([r[f'{col}_unique_ratio'] for r in all_results_full])
        status = "✅" if avg_ratio > 50 else "⚠️" if avg_ratio > 10 else "❌"
        print(f"  {col:12s}: {avg_ratio:6.2f}% {status}")

    # 平均连续重复比例
    print(f"\n各signal列平均最长连续重复比例:")
    for col in ['signal_0', 'signal_1', 'signal_2']:
        avg_ratio = np.mean([r[f'{col}_consecutive_dup_ratio'] for r in all_results_full])
        status = "✅" if avg_ratio < 10 else "⚠️" if avg_ratio < 30 else "❌"
        print(f"  {col:12s}: {avg_ratio:6.2f}% {status}")

    # 平均重复行比例
    avg_dup_ratio = np.mean([r['duplicate_signal_ratio'] for r in all_results_full])
    status = "✅" if avg_dup_ratio < 10 else "⚠️" if avg_dup_ratio < 30 else "❌"
    print(f"\n平均signal行重复比例: {avg_dup_ratio:.2f}% {status}")

    # 建议
    print("\n\n" + "=" * 80)
    print("💡 建议:")
    print("=" * 80)

    time_avg = np.mean([r['time_unique_ratio'] for r in all_results_full])
    if time_avg < 5:
        print("❌ Time(s)列几乎没有信息量，建议完全移除")

    print("❌ Torque列应该移除（根据您的说明，不应作为特征）")

    signal_avg = np.mean([np.mean([r[f'signal_{i}_unique_ratio'] for i in range(3)])
                          for r in all_results_full])
    if signal_avg < 30:
        print("⚠️ signal列有大量重复值，可能需要:")
        print("   1. 检查数据采集是否正常")
        print("   2. 考虑去除连续重复的行")
        print("   3. 或使用差分(diff)而不是原始值进行训练")

    print("\n建议的特征配置:")
    print("  INPUT_DIM = 3  # 仅使用 signal_0, signal_1, signal_2")
    print("  USE_ALL_FEATURES = False  # 不使用Time和Torque")

    print("=" * 80)


if __name__ == '__main__':
    main()
