import json
import os
import sys


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, obj):
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def flatten_to_38(src: dict) -> dict:
    """
    Flatten the nested question_lib (top-level grouped items each with multiple slots)
    into 38 top-level items, each containing exactly one slot '1'. The order is
    determined by sorting the original top-level keys numerically, then sorting
    the inner slot keys numerically.

    NOTE: This does not validate that the total number of slots equals 38.
    Callers should verify after migration as needed.
    """
    flat = {}
    new_idx = 1
    for item_k in sorted(src.keys(), key=lambda x: int(x)):
        slots = src[item_k]
        for slot_k in sorted(slots.keys(), key=lambda x: int(x)):
            slot = slots[slot_k]
            flat[str(new_idx)] = {"1": slot}
            new_idx += 1
    return flat


def main():
    if len(sys.argv) < 3:
        print("Usage: python migrate_question_lib_to_38.py <src_json> <dst_json>")
        sys.exit(2)
    src_path = sys.argv[1]
    dst_path = sys.argv[2]

    src = load_json(src_path)
    dst = flatten_to_38(src)

    # Minimal verification
    top_level = len(dst)
    total_slots = sum(len(v) for v in dst.values())
    if top_level != 38:
        print(f"WARNING: top-level items = {top_level}, expected 38")
    if not all(list(v.keys()) == ["1"] for v in dst.values()):
        print("WARNING: some items do not have exactly one slot '1'")

    save_json(dst_path, dst)
    print(f"Saved migrated question lib to: {dst_path}")
    print(f"Top-level items: {top_level}; Total slots: {total_slots}")


if __name__ == "__main__":
    main()


