import os
import json
import re

def build_data_index(save_dir, json_name="data.json"):
    img_dir = os.path.join(save_dir, "images")

    assert os.path.isdir(img_dir), f"Missing directory: {img_dir}"

    # Match filenames like:
    # source_0__<target_view>.png
    # target__<target_view>.png
    pattern = re.compile(r"^(?:source_\d+|target)__([^\.]+)\.png$")

    data = {}

    for fname in os.listdir(img_dir):
        m = pattern.match(fname)
        if not m:
            continue

        target_view_name = m.group(1)
        data[target_view_name] = 1  # value intentionally unused

    json_path = os.path.join(save_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[✓] Data index written to {json_path}")
    print(f"[✓] Found {len(data)} unique samples")


if __name__ == "__main__":
    build_data_index(
        save_dir="/projects/rps/dff6142/fouheylab/dma9300/occ_project/occ/matterport_pairs/"
    )
