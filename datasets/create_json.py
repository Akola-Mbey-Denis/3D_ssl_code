import os
import json
import re

def build_data_index(save_dir, json_name="data.json"):
    img_dir = os.path.join(save_dir, "images")

    assert os.path.isdir(img_dir), f"Missing directory: {img_dir}"

    # Match prefix before _src<idx>_ or _target_
    prefix_pattern = re.compile(r"^(.*?)(?:_src\d+|_target)_")

    data = {}

    for fname in os.listdir(img_dir):
        m = prefix_pattern.match(fname)
        if not m:
            continue

        prefix = m.group(1)
        # breakpoint()
        data[prefix] = 1  # value intentionally unused

    json_path = os.path.join(save_dir, json_name)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[✓] Data index written to {json_path}")
    print(f"[✓] Found {len(data)} unique samples")


if __name__ == "__main__":
    build_data_index(
        save_dir="/projects/rps/dff6142/fouheylab/dma9300/occ_project/occ/matterportMultivew_MVDRDF"
    )
