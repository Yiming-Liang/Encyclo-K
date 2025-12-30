import os
import argparse
import glob
import re

def find_jsonl_files(folder, split):
    # 查找所有 jsonl 文件，但排除 .jsonl.tmp 文件
    files = glob.glob(os.path.join(folder, '**', '*.jsonl'), recursive=True)
    files = [f for f in files if not f.endswith('.jsonl.tmp')]
    pattern = re.compile(
        rf'^(?P<model_name>.+)_{split}_(?P<mode>.+?)(?:_(?P<index>\d+)_(?P<world_size>\d+))?\.jsonl$'
    )
    groups = dict()
    for f in files:
        fname = os.path.basename(f)
        m = pattern.match(fname)
        if not m:
            continue
        model_name = m.group('model_name')
        mode = m.group('mode')
        index = m.group('index')
        world_size = m.group('world_size')
        prefix = f"{model_name}_{split}_{mode}"
        if index and world_size:
            # 分片的文件
            groups.setdefault(prefix, dict()).setdefault(world_size, dict())[int(index)] = f
        else:
            # 非分片的文件可以略过
            pass
    return groups

def merge_jsonl(files, out_file):
    with open(out_file, 'w', encoding='utf-8') as fout:
        for f in files:
            with open(f, 'r', encoding='utf-8') as fin:
                for line in fin:
                    fout.write(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, required=True, help="要查找的文件夹")
    parser.add_argument('--split', type=str, default="knowledge_permuted_v0.random_combination.100_perbook", help="dataset name")
    parser.add_argument('--remove', action='store_true', help="合并后删除原分片文件")
    args = parser.parse_args()

    groups = find_jsonl_files(args.folder, args.split)
    for prefix in groups:
        for world_size in groups[prefix]:
            parts = groups[prefix][world_size]
            if len(parts) == int(world_size):
                out_file = os.path.join(args.folder, f"{prefix}.jsonl")
                sorted_files = [parts[i] for i in sorted(parts.keys())]
                print(f"Merging {len(sorted_files)} parts into {out_file}")
                merge_jsonl(sorted_files, out_file)
                if args.remove:
                    for f in sorted_files:
                        print(f"Removing {f}")
                        os.remove(f)
            else:
                print(f"Warning: {prefix}_*_{world_size}.jsonl parts found: {len(parts)}, expected: {world_size}")

if __name__ == "__main__":
    main()