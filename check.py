import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="检查JSONL文件中缺失特定答案格式的行。")
    parser.add_argument("--input", type=str, default=r"out\gsm8k_0p1b.jsonl", help="输入文件的路径")
    args = parser.parse_args()
    TARGET_STR = r"Therefore, the answer is \(\boxed{"

    file_path = args.input

    if not os.path.exists(file_path):
        print(f"Error: 找不到文件 {file_path}")
        return

    print(f"Target: {TARGET_STR}")
    print(f"Scanning: {file_path}")
    print("-" * 30)

    count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                if not line.strip():
                    continue

                if TARGET_STR not in line:
                    line_num = index + 1
                    print(f"Line {line_num}")
                    count += 1

    except Exception as e:
        print(f"发生错误: {e}")

    print("-" * 30)
    print(f"总计缺失行数: {count}")


if __name__ == "__main__":
    main()