import os
import json
import argparse

TESTSET_JSON = "data/test_unseen_new.json"
MARKS = ["full", "UM", "UO"]


def load_testset(testset_path):
    with open(testset_path, "r") as f:
        return json.load(f)


def get_map_name(traj_name, testset):
    ori_traj = traj_name.replace("oracle_", "").replace("success_", "")
    for traj_path in testset:
        if ori_traj in traj_path:
            return traj_path.split("/")[-2]
    return None


def analyse_eval_log(result_dir, testset_path=TESTSET_JSON):
    testset = load_testset(testset_path)

    keys = ["collision", "oracle_success", "success", "spl", "sst", "path_length", "distance_to_end"]
    acc = {k: {m: 0 for m in MARKS} for k in keys}
    total_cnt = {m: 0 for m in MARKS}

    for traj in os.listdir(result_dir):
        traj_path = os.path.join(result_dir, traj)
        if not os.path.isdir(traj_path):
            continue
        map_name = get_map_name(traj, testset)
        if map_name is None:
            print(f"No matching testset entry for {traj}")
            continue
        state_path = os.path.join(traj_path, "state_log.json")
        if not os.path.exists(state_path):
            print(f"State log not found: {state_path}")
            continue
        with open(state_path, "r") as f:
            s = json.load(f)

        subset_mark = "UM" if map_name == "ModularPark" else "UO"
        for mark in ["full", subset_mark]:
            for k in keys:
                if k == "collision":
                    acc[k][mark] += 1 if (s["collision"] and not s["success"]) else 0
                elif k in ("oracle_success", "success"):
                    acc[k][mark] += 1 if s[k] else 0
                else:
                    acc[k][mark] += s[k]
            total_cnt[mark] += 1

    for m in MARKS:
        if total_cnt[m] == 0:
            print(f"Warning: no trajectories found for subset '{m}'")
            return

    display = [
        ("SR",  "success",        100),
        ("OSR", "oracle_success", 100),
        ("CR",  "collision",      100),
        ("SPL", "spl",            100),
        ("SST", "sst",            100),
        ("APL", "path_length",      1),
        ("NE",  "distance_to_end",  1),
    ]

    sep = "-" * 46
    print(sep)
    print(f'{"":^10}|{"Full":^10}|{"UM":^10}|{"UO":^10}')
    print(f'{"Total Cnt":^10}|{total_cnt["full"]:^10}|{total_cnt["UM"]:^10}|{total_cnt["UO"]:^10}')
    for label, key, scale in display:
        vals = {m: acc[key][m] / total_cnt[m] * scale for m in MARKS}
        print(f'{label:^10}|{vals["full"]:^10.2f}|{vals["UM"]:^10.2f}|{vals["UO"]:^10.2f}')
    print(sep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_dir", type=str, default="./output")
    parser.add_argument("--testset", type=str, default=TESTSET_JSON)
    args = parser.parse_args()
    analyse_eval_log(args.result_dir, args.testset)
