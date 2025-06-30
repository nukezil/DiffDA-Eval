import re
import os
import shutil
from datetime import datetime
import argparse

def create_symlinks_to_latest(src_dir):
    pattern = re.compile(r"^(.*)_(\d{8})$")
    latest_dirs = {}

    for name in os.listdir(src_dir):
        full_path = os.path.join(src_dir, name)
        if not os.path.isdir(full_path):
            continue

        match = pattern.match(name)
        if match:
            prefix, timestamp = match.groups()
            try:
                time_obj = datetime.strptime(timestamp, "%m%d%H%M")
            except ValueError:
                continue

            if prefix not in latest_dirs or time_obj > latest_dirs[prefix][0]:
                latest_dirs[prefix] = (time_obj, full_path)

    for prefix, (_, real_path) in latest_dirs.items():
        symlink_path = os.path.join(src_dir, prefix)
        if os.path.islink(symlink_path) or os.path.exists(symlink_path):
            print(f"Delete existing paths: {symlink_path}")
            if os.path.islink(symlink_path):
                os.unlink(symlink_path)
            else:
                shutil.rmtree(symlink_path)
        print(f"Create symlink: {symlink_path} -> {real_path}")
        os.symlink(real_path, symlink_path)

if __name__  == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--func", type=str, default="symlinks")
    parser.add_argument("--src_dir", type=str, default=None)
    args = parser.parse_args()
    if args.func == "symlinks":
        src_dir = args.src_dir
        create_symlinks_to_latest(src_dir)