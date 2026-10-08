#!/usr/bin/env python3
"""Collect the ViT lane dataset over every road_NN map, unattended.

  source /opt/ros/humble/setup.bash && source ~/sim_gazebo/install/setup.bash
  /usr/bin/python3 collect_all.py                       # every road_NN map
  /usr/bin/python3 collect_all.py road_03 road_07       # just these
  /usr/bin/python3 collect_all.py --n 20 --out /tmp/x road_00   # smoke test
  /usr/bin/python3 collect_all.py --n 400 --tag avoid --avoid-only road_00 ...
        # extra avoidance-only run per map into raw/<map>__avoid/ (existing
        # frames untouched); maps with no passable obstacle are skipped
  /usr/bin/python3 collect_all.py --n 300 --tag right --avoid-only --right-pass road_93 ...
        # obstacles whose only legal pass is on the right (inner lane by the centre line)
  /usr/bin/python3 collect_all.py --n 300 --tag wrong --wrong-only road_93 ...
        # every episode starts on the wrong side and steers back (two-way maps only)
  /usr/bin/python3 collect_all.py --n 400 --tag stop --stop-only road_99 ...
        # every episode starts behind a full road block and stops (maps without one are skipped)
  /usr/bin/python3 collect_all.py --n 2000 --parallel 3
        # 3 sims at once, each fully isolated (see below)

Per map: launch the sim with Gazebo + RViz windows -> road_collect drives episodes until it has
n_samples frames and exits -> kill that map's processes -> next map.

--parallel N runs N maps at once. Each worker slot gets its own ROS_DOMAIN_ID
(topics of different sims never meet: no car is steered, teleported or filmed
by another worker's collector) and its own GAZEBO_MASTER_URI port, launches
with require_clean_runtime:=false, and kills only its own process groups.
Each map writes only its own folder and logs, so the data cannot mix. Sims
run on their own simulated clocks: a busy machine makes them slower, never
wrong. RAM guard: a worker only launches its sim while at least --min-free-gb
is available (each instance with windows takes ~2 GB), so --parallel is an
upper bound -- it never pushes the machine into the OOM killer, which would
pick the biggest process (e.g. someone else's training run).

Resumable: a map whose poses.csv already holds n_samples frames is skipped, a
partial one is redone from scratch, so after an interruption just run it
again. Refuses to start while another sim is running.
Logs: <out>/_logs/<map>_sim.log and <map>_collect.log.
"""

import argparse
import concurrent.futures
import glob
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, '..', '..', 'config', 'lane_assist_params.yaml')
SIM_PROCS = ['gzserver', 'gzclient', 'rviz2', 'robot_state_publisher', 'map_publisher',
             'dynamics_plant', 'dynamic_obstacles', 'lpv_mpc_node', 'ackermann_to_twist',
             'pose_reset', 'spawn_entity', 'road_collect']


def running():
    out = subprocess.run(['ps', '-eo', 'pid,args'], capture_output=True, text=True).stdout
    return [l for l in out.splitlines()[1:]
            if any(p in l for p in SIM_PROCS) or 'sim_launch.py' in l]


def mem_free_gb():
    for line in open('/proc/meminfo'):
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024 ** 2
    return 0.0


def group_alive(pgid):
    out = subprocess.run(['ps', '-eo', 'pgid,pid'], capture_output=True, text=True).stdout
    return [l.split()[1] for l in out.splitlines()[1:] if l.split()[0] == str(pgid)]


def kill_group(proc):
    """Stop one launched process and everything it started (its own process
    group): SIGINT, then SIGKILL -- gzserver and robot_state_publisher often
    ignore SIGINT. Never touches other workers' sims."""
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        pass
    for _ in range(20):
        if not group_alive(proc.pid):
            return
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        time.sleep(0.5)
    raise RuntimeError(f'could not kill process group {proc.pid}: {group_alive(proc.pid)}')


def frames(d):
    """Frames in a run's poses.csv. Labels are rebuilt from the pose alone,
    so runs from earlier collector versions count as done; only the very
    first format (no episodes) is redone."""
    p = os.path.join(d, 'poses.csv')
    if not os.path.exists(p) or 'episode' not in open(p).readline():
        return 0
    return sum(1 for _ in open(p)) - 1


def main():
    cfg = yaml.safe_load(open(PARAMS))['road_collect']['ros__parameters']
    ap = argparse.ArgumentParser()
    ap.add_argument('maps', nargs='*')
    ap.add_argument('--n', type=int, default=int(cfg['n_samples']))
    ap.add_argument('--out', default=os.path.expanduser(cfg['out_dir']))
    ap.add_argument('--timeout', type=float, default=900.0, help='per map [s]')
    ap.add_argument('--tag', default='', help='extra run per map into <map>__<tag>/')
    ap.add_argument('--avoid-only', action='store_true',
                    help='every episode starts behind a passable obstacle')
    ap.add_argument('--right-pass', action='store_true',
                    help='with --avoid-only: every one behind an obstacle whose only legal pass is on the right')
    ap.add_argument('--stop-only', action='store_true',
                    help='every episode starts behind a FULL road block and brakes to a stop '
                         '(maps without one are skipped)')
    ap.add_argument('--wrong-only', action='store_true',
                    help='every episode starts on the wrong side of the road and steers back '
                         '(two-way maps only; one-way maps are skipped)')
    ap.add_argument('--parallel', type=int, default=1, help='maps collected at once (isolated sims)')
    ap.add_argument('--min-free-gb', type=float, default=3.0,
                    help='launch a sim only while this much RAM is available')
    ap.add_argument('--no-rviz', action='store_true', help='Gazebo window only (saves ~0.4 GB per sim)')
    args = ap.parse_args()
    maps = args.maps or sorted(os.path.basename(f)[:-4] for f in
                               glob.glob(os.path.join(os.path.expanduser(cfg['truth_dir']), 'road_[0-9]*.npz')))
    if running():
        sys.exit('a sim is already running, stop it first:\n' + '\n'.join(running()))
    logs = os.path.join(args.out, '_logs')
    os.makedirs(logs, exist_ok=True)
    env = dict(os.environ, PATH='/usr/bin:' + os.environ.get('PATH', ''))
    summary = []
    t_all = time.time()
    if args.wrong_only:
        sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..')))
        from lane_assist.eval_tools import Truth
        truth_dir = os.path.expanduser(cfg['truth_dir'])
        have = [m for m in maps if eval(str(np.load(os.path.join(truth_dir, m + '.npz'))['meta']))['road'] == 'two_way']
        if len(have) < len(maps):
            print(f'skipping {len(maps) - len(have)} one-way / divided maps (no oncoming lane to recover from): '
                  f'{sorted(set(maps) - set(have))}')
        maps = have
    if args.stop_only:
        sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..')))
        from lane_assist.eval_tools import Truth
        have = [m for m in maps if any(t < 0 for *_, t in Truth(m).maneuvers())]
        if len(have) < len(maps):
            print(f'skipping {len(maps) - len(have)} maps with no full road block: {sorted(set(maps) - set(have))}')
        maps = have
    if args.avoid_only:
        sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..')))
        from lane_assist.eval_tools import Truth
        have = [m for m in maps if any(t >= 0 and (not args.right_pass or Truth(m).passes_right(k, t))
                                       for k, _, _, t in Truth(m).maneuvers())]
        if len(have) < len(maps):
            print(f'skipping {len(maps) - len(have)} maps with no passable obstacle: '
                  f'{sorted(set(maps) - set(have))}')
        maps = have
    todo = []
    for name in maps:
        d = os.path.join(args.out, name + ('__' + args.tag if args.tag else ''))
        if frames(d) >= args.n:
            summary.append((name, frames(d), 'skipped (done)'))
        else:
            todo.append((name, d))
    slots = queue.Queue()
    for k in range(max(1, args.parallel)):
        slots.put(k)
    lock = threading.Lock()
    count = [0]

    def run_map(name, d):
        slot = slots.get()
        try:
            with lock:                              # one launch at a time, after memory settles
                waited = False
                while mem_free_gb() < args.min_free_gb:
                    if not waited:
                        print(f'    {name}: waiting for RAM ({mem_free_gb():.1f} GB free, '
                              f'need {args.min_free_gb:.1f})', flush=True)
                        waited = True
                    time.sleep(10)
            wenv = dict(env, ROS_DOMAIN_ID=str(40 + slot),
                        GAZEBO_MASTER_URI=f'http://localhost:{11345 + slot}')
            shutil.rmtree(d, ignore_errors=True)
            t0 = time.time()
            with lock:
                count[0] += 1
                print(f'[{count[0]}/{len(todo)}] {name} ... (slot {slot})', flush=True)
                sim = subprocess.Popen(
                    ['ros2', 'launch', 'f1tenth_gym_gazebo', 'sim_launch.py', f'map_name:={name}',
                     'gui:=true', f'rviz:={"false" if args.no_rviz else "true"}']
                    + (['require_clean_runtime:=false'] if args.parallel > 1 else []),
                    stdout=open(os.path.join(logs, name + '_sim.log'), 'w'),
                    stderr=subprocess.STDOUT, env=wenv, start_new_session=True)
                time.sleep(25)                      # let it reach full size before the next launch
            col = subprocess.Popen(
                ['ros2', 'run', 'lane_assist', 'road_collect', '--ros-args',
                 '--params-file', PARAMS, '-p', f'map_name:={name}',
                 '-p', f'n_samples:={args.n}', '-p', f'out_dir:={args.out}']
                # ROS rejects an empty override ('-p run_tag:='), so only pass a real tag
                + (['-p', f'run_tag:={args.tag}'] if args.tag else [])
                + (['-p', 'obstacle_start_frac:=1.0'] if args.avoid_only else [])
                + (['-p', 'right_pass_frac:=1.0'] if args.right_pass else [])
                + (['-p', 'stop_frac:=1.0', '-p', 'obstacle_start_frac:=0.0'] if args.stop_only else [])
                + (['-p', 'obstacle_start_frac:=0.0', '-p', 'stop_frac:=0.0', '-p', 'wrong_frac:=1.0']
                   if args.wrong_only else []),
                stdout=open(os.path.join(logs, name + '_collect.log'), 'w'),
                stderr=subprocess.STDOUT, env=wenv, start_new_session=True)
            status = 'ok'
            try:
                col.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                status = 'TIMEOUT'
            finally:
                kill_group(col)
                kill_group(sim)
            got = frames(d)
            if got < args.n and status == 'ok':
                status = 'INCOMPLETE'
            with lock:
                summary.append((name, got, f'{status} {time.time() - t0:.0f}s'))
                print(f'    {name}: {got} frames, {summary[-1][2]}', flush=True)
        finally:
            slots.put(slot)

    with concurrent.futures.ThreadPoolExecutor(max(1, args.parallel)) as pool:
        for f in [pool.submit(run_map, n, d) for n, d in todo]:
            f.result()
    if running():
        print('WARNING: processes left over:\n' + '\n'.join(running()))
    print(f'\ndone in {(time.time() - t_all) / 60:.1f} min -> {args.out}')
    for name, got, st in sorted(summary):
        print(f'  {name}: {got:4d} frames  {st}')


if __name__ == '__main__':
    main()
