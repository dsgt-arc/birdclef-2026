#!/usr/bin/env python3
"""
check_gpus.py - GPU partition availability and queue status on PACE HPC.

Usage:
  python3 check_gpus.py [-u USERNAME] [-p PARTITIONS]

Examples:
  python3 check_gpus.py
  python3 check_gpus.py -u rbrock8
  python3 check_gpus.py -p gpu-a100,gpu-h100 -u rbrock8
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime

DEFAULT_PARTITIONS = "gpu-a100,gpu-h100,gpu-h200,gpu-rtx6000"

# ── ANSI helpers ──────────────────────────────────────────────────────────────
USE_COLOR = sys.stdout.isatty()


def _c(code, text):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def bold(t):
    return _c("1", t)


def dim(t):
    return _c("2", t)


def green(t):
    return _c("92", t)


def yellow(t):
    return _c("93", t)


def red(t):
    return _c("91", t)


def cyan(t):
    return _c("96", t)


def white(t):
    return _c("97", t)


def pct_color(pct, *, lo=60, hi=85):
    """Color a percentage string: green < lo, yellow lo–hi, red > hi."""
    if pct is None:
        return "N/A"
    s = f"{pct:.0f}%"
    if pct < lo:
        return green(s)
    if pct <= hi:
        return yellow(s)
    return red(s)


def gpu_color(n):
    if n > 0:
        return bold(green(str(n)))
    return dim(str(n))


def ahead_color(n):
    if n <= 5:
        return green(f"{n:.1f}")
    if n <= 20:
        return yellow(f"{n:.1f}")
    return red(f"{n:.1f}")


# ── Box-drawing table helpers ─────────────────────────────────────────────────
def _row(cells, widths, sep="│"):
    parts = [f" {str(c):<{widths[i]}} " for i, c in enumerate(cells)]
    return sep + sep.join(parts) + sep


def _colored_row(cells, widths, sep="│"):
    """Like _row but cells may contain ANSI codes — pad by visible length."""

    def pad(s, w):
        visible = re.sub(r"\033\[[0-9;]*m", "", s)
        return s + " " * max(0, w - len(visible))

    parts = [f" {pad(str(c), widths[i])} " for i, c in enumerate(cells)]
    return sep + sep.join(parts) + sep


def _divider(widths, left="├", mid="┼", right="┤", fill="─"):
    return left + mid.join(fill * (w + 2) for w in widths) + right


def _top(widths):
    return "┌" + "┬".join("─" * (w + 2) for w in widths) + "┐"


def _bottom(widths):
    return "└" + "┴".join("─" * (w + 2) for w in widths) + "┘"


def print_table(headers, rows, colored_rows=None):
    """
    Print a Unicode box table. colored_rows is a parallel list to rows where
    each cell may contain ANSI codes; rows is used for width calculation only.
    """
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    print(_top(widths))
    print(_colored_row([bold(white(h)) for h in headers], widths))
    print(_divider(widths))
    for i, row in enumerate(rows):
        crow = colored_rows[i] if colored_rows else row
        print(_colored_row(crow, widths))
    print(_bottom(widths))


# ── Data fetching ─────────────────────────────────────────────────────────────
def run(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout


def parse_pace_summary(output):
    info = {}
    for line in output.splitlines():
        if "Number of Nodes" in line:
            info["nodes"] = line.split(":", 1)[1].strip()
        elif "Number of Cores" in line:
            info["cores"] = line.split(":", 1)[1].strip()
        elif "Amount of Memory" in line:
            # pace-check-queue labels columns as "Used/Total" but "Total" is
            # actually the remaining FREE memory, making the shown % = used/free.
            # Parse the raw numbers to compute the correct utilization.
            val = line.split(":", 1)[1].strip()  # e.g. "21070/3091 (681.66%)"
            m = re.match(r"([\d.]+)/([\d.]+)", val)
            if m:
                used = float(m.group(1))
                free = float(m.group(2))
                total = used + free
                info["mem_used_gb"] = used
                info["mem_total_gb"] = total
                info["mem_pct"] = (used / total * 100) if total else 0
            else:
                info["memory"] = val  # fallback
    return info


def parse_percent(s):
    if "(" in s and "%" in s:
        try:
            return float(s.split("(")[1].split("%")[0])
        except ValueError:
            pass
    return None


def get_node_availability(partitions):
    results = {}
    for p in partitions:
        out = run(f"pace-check-queue {p}")
        results[p] = parse_pace_summary(out)
    return results


def parse_gpu_count(gres_str):
    m = re.search(r"gpu:[^:]+:(\d+)", gres_str)
    return int(m.group(1)) if m else 0


def get_gpu_info(partitions):
    results = {}
    for p in partitions:
        out = run(f"sinfo -p {p} -N --Format='nodelist:40,gres:50,gresused:50' --noheader")
        total_gpus = 0
        total_free = 0
        max_free = 0
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            total = parse_gpu_count(parts[1])
            used = parse_gpu_count(parts[2])
            free = max(0, total - used)
            total_gpus += total
            total_free += free
            max_free = max(max_free, free)
        results[p] = {"total_gpus": total_gpus, "total_free": total_free, "max_free": max_free}
    return results


def get_jobs(partitions):
    partition_str = ",".join(partitions)
    out = run(f"squeue -p {partition_str} --json")
    try:
        data = json.loads(out)
        return data.get("jobs", [])
    except json.JSONDecodeError:
        print("Error: failed to parse squeue JSON output.", file=sys.stderr)
        sys.exit(1)


COUNTABLE_REASONS = {"Priority", "Resources"}


def job_state(job):
    s = job.get("job_state", "UNKNOWN")
    return s[0] if isinstance(s, list) else s


def is_competing(job):
    return (
        job.get("qos", "").lower() == "inferno" and job.get("state_reason", "") in COUNTABLE_REASONS
    )


def weighted_counts(jobs, target_set):
    counts = {p: {"pending": 0.0, "running": 0.0} for p in target_set}
    for j in jobs:
        state = job_state(j)
        part_str = j.get("partition", "")
        if state == "PENDING":
            if not is_competing(j):
                continue
            job_parts = set(part_str.split(","))
            overlap = job_parts & target_set
            if not overlap:
                continue
            weight = 1.0 / len(job_parts)
            for p in overlap:
                counts[p]["pending"] += weight
        elif state == "RUNNING":
            if part_str in target_set:
                counts[part_str]["running"] += 1.0
    return counts


def get_user_jobs(username):
    """Get all jobs for a user (any partition)."""
    out = run(f"squeue -u {username} --json")
    try:
        data = json.loads(out)
        return data.get("jobs", [])
    except json.JSONDecodeError:
        return []


def _elapsed_str(job):
    """Compute elapsed time string from start_time."""
    start = job.get("start_time", {})
    start_ts = start.get("number", 0) if isinstance(start, dict) else 0
    if not start_ts:
        return "—"
    import time

    elapsed = max(0, int(time.time()) - start_ts)
    h, rem = divmod(elapsed, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h{m:02d}m"


def _parse_gpu_type(job):
    """Extract GPU type from tres_alloc_str (e.g. 'gres/gpu:v100=1' → 'v100')."""
    tres = job.get("tres_alloc_str", "")
    m = re.search(r"gres/gpu:([^=]+)=(\d+)", tres)
    if m:
        return f"{m.group(1)}×{m.group(2)}" if m.group(2) != "1" else m.group(1)
    # Check if any gpu allocated at all
    if "gres/gpu=" in tres:
        return "gpu"
    return ""


def jobs_ahead(jobs, username, target_set):
    user_pending = [
        j
        for j in jobs
        if j.get("user_name") == username
        and job_state(j) == "PENDING"
        and (set(j.get("partition", "").split(",")) & target_set)
    ]
    other_pending = [
        j
        for j in jobs
        if j.get("user_name") != username and job_state(j) == "PENDING" and is_competing(j)
    ]
    results = []
    for uj in sorted(user_pending, key=lambda x: x.get("job_id", 0)):
        uid = uj.get("job_id", 0)
        u_parts = set(uj.get("partition", "").split(",")) & target_set
        if not u_parts:
            continue
        # Count weighted jobs ahead per partition, then take the min.
        # If you request A100,H100 and there are 10.5 ahead on A100 but
        # 0.5 on H100, you'd get the H100 first — so 0.5 ahead, not 11.
        per_part = dict.fromkeys(u_parts, 0.0)
        for oj in other_pending:
            if oj.get("job_id", 0) >= uid:
                continue
            o_parts = set(oj.get("partition", "").split(","))
            overlap = u_parts & o_parts
            if not overlap:
                continue
            weight = 1.0 / len(o_parts)
            for p in overlap:
                per_part[p] += weight
        ahead = min(per_part.values())
        results.append(
            {
                "job_id": uid,
                "name": uj.get("name", ""),
                "partition": uj.get("partition", ""),
                "ahead": ahead,
            }
        )
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Check GPU partition availability and queue depth on PACE HPC."
    )
    parser.add_argument("-u", "--user", help="Username to check queue position for (e.g. rbrock8)")
    parser.add_argument(
        "-p",
        "--partitions",
        default=DEFAULT_PARTITIONS,
        help=f"Comma-separated partitions (default: {DEFAULT_PARTITIONS})",
    )
    args = parser.parse_args()

    partitions = [p.strip() for p in args.partitions.split(",") if p.strip()]
    target_set = set(partitions)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{bold(cyan('  PACE GPU Status'))}  {dim(now)}")

    # ── Section 1: Node availability ─────────────────────────────────────────
    print(f"\n{bold('Node Availability')}  {dim('(PACE cache updated every 15 min)')}")
    availability = get_node_availability(partitions)
    gpu_info = get_gpu_info(partitions)

    headers = [
        "Partition",
        "Nodes acc/tot",
        "Cores used",
        "Mem used",
        "Total GPUs",
        "Free GPUs",
        "Max/node",
    ]
    rows, colored_rows = [], []

    for p in partitions:
        info = availability[p]
        nodes = info.get("nodes", "N/A")
        cores = info.get("cores", "N/A")

        nodes_str = nodes.split("(")[0].strip() if nodes != "N/A" else "N/A"
        acc, tot = None, None
        if "/" in nodes_str:
            try:
                acc, tot = (int(x.strip()) for x in nodes_str.split("/"))
            except ValueError:
                pass

        cores_pct = parse_percent(cores)
        mem_pct = info.get("mem_pct")
        ginfo = gpu_info.get(p, {})
        tg = ginfo.get("total_gpus", 0)
        tf = ginfo.get("total_free", 0)
        mf = ginfo.get("max_free", 0)

        rows.append(
            [
                p,
                nodes_str,
                f"{cores_pct:.0f}%" if cores_pct else cores,
                f"{mem_pct:.0f}%" if mem_pct is not None else "N/A",
                str(tg),
                str(tf),
                str(mf),
            ]
        )

        if acc is not None and tot is not None:
            ratio = acc / tot if tot else 0
            node_c = (
                green(nodes_str)
                if ratio == 1.0
                else yellow(nodes_str)
                if ratio >= 0.5
                else red(nodes_str)
            )
        else:
            node_c = nodes_str

        colored_rows.append(
            [
                bold(p),
                node_c,
                pct_color(cores_pct),
                pct_color(mem_pct),
                dim(str(tg)),
                gpu_color(tf),
                gpu_color(mf),
            ]
        )

    print_table(headers, rows, colored_rows)

    # ── Section 2: Queue depth ───────────────────────────────────────────────
    print(f"\n{bold('Queue Depth')}  {dim('(inferno · Priority + Resources only)')}")
    jobs = get_jobs(partitions)
    counts = weighted_counts(jobs, target_set)

    headers = ["Partition", "~Pending", "~Running"]
    rows, colored_rows = [], []
    for p in partitions:
        c = counts[p]
        pending_s = f"{c['pending']:.1f}"
        running_s = f"{c['running']:.1f}"
        rows.append([p, pending_s, running_s])
        colored_rows.append([bold(p), yellow(pending_s), cyan(running_s)])

    print_table(headers, rows, colored_rows)

    # ── Section 3: User's jobs ───────────────────────────────────────────────
    if args.user:
        # Running jobs (all partitions, not just GPU)
        all_user_jobs = get_user_jobs(args.user)
        running = [j for j in all_user_jobs if job_state(j) == "RUNNING"]

        print(f"\n{bold('Your Running Jobs')}  {dim(f'({args.user})')}")
        if not running:
            print(f"  {dim('No running jobs.')}")
        else:
            headers = ["JobID", "Name", "Partition", "Node", "Runtime", "GPU"]
            rows, colored_rows = [], []
            for j in running:
                jid = str(j.get("job_id", ""))
                name = j.get("name", "")
                part = j.get("partition", "")
                node = j.get("nodes", "?")
                elapsed = _elapsed_str(j)
                gpu = _parse_gpu_type(j)
                rows.append([jid, name, part, node, elapsed, gpu])
                colored_rows.append(
                    [
                        dim(jid),
                        white(name),
                        cyan(part),
                        dim(node),
                        green(elapsed),
                        yellow(gpu) if gpu else dim("—"),
                    ]
                )
            print_table(headers, rows, colored_rows)

        # Pending jobs
        pending_jobs = jobs_ahead(jobs, args.user, target_set)
        if pending_jobs:
            print(f"\n{bold('Your Pending Jobs')}  {dim(f'({args.user})')}")
            headers = ["JobID", "Name", "Partition(s) Requested", "~Jobs Ahead"]
            rows, colored_rows = [], []
            for j in pending_jobs:
                ahead_s = f"{j['ahead']:.1f}"
                rows.append([str(j["job_id"]), j["name"], j["partition"], ahead_s])
                colored_rows.append(
                    [
                        dim(str(j["job_id"])),
                        white(j["name"]),
                        cyan(j["partition"]),
                        ahead_color(j["ahead"]),
                    ]
                )
            print_table(headers, rows, colored_rows)

    print()


if __name__ == "__main__":
    main()
