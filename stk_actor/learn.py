from pathlib import Path
from pystk2_gymnasium import AgentSpec
from functools import partial
import torch
from bbrl.agents.gymnasium import ParallelGymAgent, make_env

# Note the use of relative imports
from .actors import Actor
from .pystk_actor import env_name, get_wrappers, player_name
from .config import ppoconfig
from .sequenceBuffer import sequenceReplayBuffer
from .types import Transition, Batch

try:
    from tqdm.auto import tqdm
except Exception:
    from tqdm import tqdm

import time
from collections import deque, defaultdict
import re

MAX_CKPTS = 10
ckpt_history = deque(maxlen=MAX_CKPTS)

# 回溯次数统计：同一个ckpt被rollback超过2次就跳过
rollback_count = defaultdict(int)
MAX_ROLLBACK_PER_CKPT = 2


def _params_are_finite(actor) -> bool:
    for p in actor.parameters():
        if p is None:
            continue
        if not torch.isfinite(p).all():
            return False
    return True


def _load_ckpt_into_actor(actor, ckpt_path: Path):
    """Load ckpt and check finiteness. Return (step or None)."""
    if not ckpt_path.exists():
        return None
    data = torch.load(ckpt_path, map_location="cpu")
    actor.load_state_dict(data["actor_state_dict"])
    if not _params_are_finite(actor):
        return None
    return int(data.get("global_step", 0))


def _rollback_latest_first_then_stable(actor, latest_ckpt: Path):
    """
    你要的策略：
    1) 先回 latest.pt
       - 如果 latest NaN/Inf 或者同一个latest回溯超过2次 -> 跳过
    2) 再回 stable池，从新到旧尝试
       - stable若坏（NaN/Inf）或缺失 -> 从history移除并继续尝试更旧
    """
    # ---- 1) try latest first ----
    if latest_ckpt.exists():
        rollback_count[latest_ckpt] += 1
        if rollback_count[latest_ckpt] <= MAX_ROLLBACK_PER_CKPT:
            print(f"[ROLLBACK] Try latest checkpoint {latest_ckpt.name} (used {rollback_count[latest_ckpt]}/{MAX_ROLLBACK_PER_CKPT})")
            step = _load_ckpt_into_actor(actor, latest_ckpt)
            if step is not None:
                print(f"[ROLLBACK] Success: {latest_ckpt.name} (global_step={step})")
                return step
            else:
                print(f"[ROLLBACK] Latest {latest_ckpt.name} is corrupted (NaN/Inf). Will try stable.")
        else:
            print(f"[ROLLBACK] Latest {latest_ckpt.name} used too many times, skip -> try stable")

    # ---- 2) try stable history newest -> oldest ----
    candidates = list(ckpt_history)[::-1]
    for ckpt_path in candidates:
        rollback_count[ckpt_path] += 1
        if rollback_count[ckpt_path] > MAX_ROLLBACK_PER_CKPT:
            print(f"[ROLLBACK] {ckpt_path.name} used {rollback_count[ckpt_path]} times, skip -> try older")
            continue

        if not ckpt_path.exists():
            print(f"[ROLLBACK] Missing {ckpt_path.name}, remove from history")
            try:
                ckpt_history.remove(ckpt_path)
            except ValueError:
                pass
            continue

        print(f"[ROLLBACK] Try stable checkpoint {ckpt_path.name} (used {rollback_count[ckpt_path]}/{MAX_ROLLBACK_PER_CKPT})")
        step = _load_ckpt_into_actor(actor, ckpt_path)
        if step is not None:
            print(f"[ROLLBACK] Success: {ckpt_path.name} (global_step={step})")
            return step
        else:
            print(f"[ROLLBACK] {ckpt_path.name} is corrupted (NaN/Inf). Drop from history and try older.")
            try:
                ckpt_history.remove(ckpt_path)
            except ValueError:
                pass
            # 磁盘文件建议不删；你要删再自行打开
            # try:
            #     ckpt_path.unlink()
            # except Exception:
            #     pass

    print("[ROLLBACK] No usable checkpoints left.")
    return None


def _save_all(actor, global_step: int, tag: str = "latest"):
    """Save evaluation file + latest checkpoint."""
    mod_path = Path(__file__).resolve().parent
    ckpt_dir = mod_path / "checkpoints"
    mod_path.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    eval_path = mod_path / "pystk_actor.pth"
    torch.save(actor.state_dict(), eval_path)

    ckpt = {
        "global_step": global_step,
        "actor_state_dict": actor.state_dict(),
        "time": time.time(),
    }
    torch.save(ckpt, ckpt_dir / f"{tag}.pt")
    print(f"[CHECKPOINT] saved eval={eval_path.name}, ckpt={tag}.pt at step={global_step}")

def _save_milestone_pth(actor, global_step: int, every: int = 51200):
    """
    Save an extra .pth (state_dict only) every `every` steps.
    Not overwritten: one file per milestone step.
    """
    if global_step % every != 0:
        return

    mod_path = Path(__file__).resolve().parent
    out_dir = mod_path / "milestones"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"pystk_actor_step_{global_step:07d}.pth"

    # 如果已经存在（比如重启后又跑到同一步），就不重复写
    if out_path.exists():
        return

    torch.save(actor.state_dict(), out_path)
    print(f"[MILESTONE] saved {out_path.relative_to(mod_path)}")


def _save_stable(actor, global_step: int):
    """Save stable_{step}.pt, keep at most 5 on disk + in history."""
    mod_path = Path(__file__).resolve().parent
    ckpt_dir = mod_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stable_path = ckpt_dir / f"stable_{global_step}.pt"

    # 已在history：覆盖写，不重复append
    if stable_path in ckpt_history:
        torch.save(
            {"global_step": global_step, "actor_state_dict": actor.state_dict(), "time": time.time()},
            stable_path,
        )
        print(f"[STABLE] updated {stable_path.name} (history={len(ckpt_history)}/{MAX_CKPTS})")
        return

    # 满了：删最旧（磁盘+history）
    if len(ckpt_history) == ckpt_history.maxlen:
        old_path = ckpt_history.popleft()
        while old_path in ckpt_history:
            ckpt_history.remove(old_path)
        try:
            if old_path.exists():
                old_path.unlink()
                print(f"[STABLE] removed old {old_path.name}")
        except Exception as e:
            print("[STABLE] remove old failed:", repr(e))

    # 写新stable
    torch.save(
        {"global_step": global_step, "actor_state_dict": actor.state_dict(), "time": time.time()},
        stable_path,
    )
    ckpt_history.append(stable_path)
    print(f"[STABLE] saved {stable_path.name} (history={len(ckpt_history)}/{MAX_CKPTS})")


seq_obs_keys = [
    "items_position",
    "items_type",
    "karts_position",
    "paths_distance",
    "paths_end",
    "paths_start",
    "paths_width",
]

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


if __name__ == "__main__":
    make_stkenv = partial(
        make_env,
        env_name,
        wrappers=get_wrappers(),
        render_mode=None,
        autoreset=True,
        agent=AgentSpec(use_ai=False, name=player_name),
    )

    env_agent = ParallelGymAgent(make_stkenv, 1)
    env = env_agent.envs[0]

    # raw dims
    seq_raw_dims = {key: None for key in seq_obs_keys}
    for key, space in env.observation_space.items():
        if key in seq_obs_keys:
            seq_raw_dims[key] = space.feature_space.shape

    dfconfig = ppoconfig(device, seq_raw_dims).config
    print(f"Using device: {device}")
    print(f"PPO Config: {dfconfig}")

    actor = Actor(None, None, "PPO", dfconfig, device)

    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    latest_ckpt = ckpt_dir / "latest.pt"
    stable_ckpts = list(ckpt_dir.glob("stable_*.pt"))

    def _stable_step(p: Path) -> int:
        m = re.search(r"stable_(\d+)\.pt$", p.name)
        return int(m.group(1)) if m else -1

    stable_ckpts.sort(key=_stable_step)  # old -> new

    # rebuild history from disk (keep newest 5)
    ckpt_history.clear()
    for p in stable_ckpts[-MAX_CKPTS:]:
        ckpt_history.append(p)
    print(f"[STABLE] history on start: {[x.name for x in ckpt_history]}")

    # resume: newest stable first, else latest
    resume_path = stable_ckpts[-1] if len(stable_ckpts) > 0 else (latest_ckpt if latest_ckpt.exists() else None)
    if resume_path is not None and resume_path.exists():
        data = torch.load(resume_path, map_location="cpu")
        actor.load_state_dict(data["actor_state_dict"])
        global_step = int(data.get("global_step", 0))
        print(f"[RESUME] Loaded {resume_path.name} (global_step={global_step})")
    else:
        global_step = 0
        print("[RESUME] No checkpoint found, start from scratch.")

    buffer = sequenceReplayBuffer(**dfconfig["ReplayBuffer"])
    train_config = dfconfig["training_config"]  # update_interval=500, batch_size=16 etc.

    max_total_steps = 6_000_000
    pbar = tqdm(total=max_total_steps, desc="Training Actor (steps)", initial=global_step)

    # checkpoint schedule
    save_every_steps = 2560
    stable_every_steps = 5120
    last_save_step = global_step
    NO_UPDATE_LIMIT = 10 # 连续 10 次“检查更新点”都不能更新，就自愈（约 200*512=10万步）
    no_update_streak = 0    

    # 用于显示，不再依赖 batch
    last_batch_reward = None

    obs, _ = env.reset()
    # ---- HEALTH probes ----
    prev_param_norm = None
    prev_action_eval = None
    last_update_step = None


    try:
        while global_step < max_total_steps:
            action, log_prob, _ = actor.algo.select_action(obs, eval_mode=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)

            t = Transition(
                states=obs,
                action=action,
                reward=reward,
                next_states=next_obs,
                truncated=truncated,
                terminated=terminated,
                log_prob=log_prob,
            )
            buffer.add(t)

            obs = next_obs
            global_step += 1

            # ---- update every 512 ----
            if global_step % train_config["update_interval"] == 0:
                # buffer里真实数据量：看 storage
                bs = len(buffer.storage) if hasattr(buffer, "storage") else 0

                # sequence buffer 真正能 sample 的依据：valid_starts 数量要够 batch_size
                valid = len(buffer.valid_starts) if hasattr(buffer, "valid_starts") else 0
                can_update = (valid >= train_config["batch_size"])
                
                # ---- auto-heal: track consecutive can_update=False ----
                if not can_update:
                    no_update_streak += 1
                else:
                    no_update_streak = 0

                # 可选：每隔几次提示一下，避免刷屏
                if (not can_update) and (no_update_streak % 5 == 0):
                    print(f"[HEALTH] can_update=False streak={no_update_streak} step={global_step} "
                        f"storage={len(buffer.storage) if hasattr(buffer,'storage') else 'NA'} "
                        f"valid={len(buffer.valid_starts) if hasattr(buffer,'valid_starts') else 'NA'}")

                # ---- trigger rebuild ----
                if no_update_streak >= NO_UPDATE_LIMIT:
                    print(f"[HEALTH] can_update=False streak={no_update_streak} step={global_step} -> REBUILD BUFFER")
                    _save_all(actor, global_step, tag='latest')  # 先落盘

                    buffer = sequenceReplayBuffer(**dfconfig["ReplayBuffer"])  # ✅ 重建 buffer
                    obs, _ = env.reset()                                      # ✅ reset 环境

                    no_update_streak = 0
                    continue
                                
                
                               
                if global_step % stable_every_steps == 0:
                    print(f"[HEALTH] step={global_step} can_update={can_update} bs={bs} valid={valid}")


                if can_update:
                    try:
                        batch = buffer.sample(train_config["batch_size"])
                        _ = actor.algo.update(batch)
                        no_update_streak = 0
                        # ---------------- HEALTH PROBE (after successful update) ----------------
                        last_update_step = global_step

                        # 1) 确认 update 真的在发生（每 5120 步打印一次，避免刷屏）
                        if global_step % stable_every_steps == 0:
                            print(f"[HEALTH] UPDATE ok at step={global_step} (interval={train_config['update_interval']})")

                        # 2) 参数范数变化：能证明参数确实在更新（policy+value 都在 actor.parameters() 里）
                        with torch.no_grad():
                            s2 = 0.0
                            for p in actor.parameters():
                                if p is None:
                                    continue
                                # 用float计算，避免mps上偶发精度怪问题
                                s2 += p.detach().float().pow(2).sum().item()
                            param_norm = (s2 ** 0.5)

                        if prev_param_norm is None:
                            prev_param_norm = param_norm
                        else:
                            delta = abs(param_norm - prev_param_norm)
                            if global_step % stable_every_steps == 0:
                                print(f"[HEALTH] param_norm={param_norm:.6e} delta={delta:.3e}")
                            prev_param_norm = param_norm
                        # -----------------------------------------------------------------------

                        # 记录 batch reward（仅用于显示）
                        if hasattr(batch, "rewards") and batch.rewards is not None:
                            last_batch_reward = batch.rewards.mean().item()

                        # update后立刻检查NaN/Inf
                        if not _params_are_finite(actor):
                            raise RuntimeError("NaN/Inf detected in parameters after update.")

                    except Exception as e:
                        print(f"[ERROR] update/explosion: {repr(e)} -> rollback")
                        _save_all(actor, global_step, tag="latest")  # 炸前先落盘

                        rb = _rollback_latest_first_then_stable(actor, latest_ckpt)
                        if rb is None:
                            raise RuntimeError("Rollback failed: no usable checkpoints.")
                        global_step = rb
                        last_save_step = global_step

                        buffer.clear()
                        obs, _ = env.reset()

                # ---- stable every 5120（你要的）----
                if global_step % stable_every_steps == 0:
                    _save_stable(actor, global_step)

                # ---- latest every 2560（你要的）----
                if global_step - last_save_step >= save_every_steps:
                    _save_all(actor, global_step, tag="latest")
                    last_save_step = global_step

                _save_milestone_pth(actor, global_step, every=51200)
                
            if terminated or truncated:
                obs, _ = env.reset()

            pbar.update(1)
            # ---------------- HEALTH PROBE (policy drift) ----------------
            # 每 5120 步，用当前 obs 做一次 eval_mode action，看策略是否有变化
            if global_step % stable_every_steps == 0:
                if last_update_step is None:
                    print(f"[HEALTH] eval_action skipped (no update yet) at step={global_step}")
                else:
                    a_eval, _, _ = actor.algo.select_action(obs, eval_mode=True)
                    print(f"[HEALTH] eval_action@{global_step} = {a_eval} (last_update={last_update_step})")
            # -------------------------------------------------------------

            if global_step % 50 == 0:
                postfix = {
                    "Buffer": f"{(buffer.size() if callable(getattr(buffer,'size',None)) else getattr(buffer,'size',0)):,}"
                }
                if last_batch_reward is not None:
                    postfix["batch_reward"] = f"{last_batch_reward:.2f}"
                postfix["reward"] = f"{reward:.2f}"
                pbar.set_postfix(postfix)

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Ctrl+C detected. Saving checkpoint...")
        _save_all(actor, global_step, tag="latest")
        if global_step % stable_every_steps == 0:
            _save_stable(actor, global_step)
        raise SystemExit(0)

    except Exception as e:
        print(f"\n[CRASH] {e}. Saving checkpoint before crash...")
        _save_all(actor, global_step, tag="latest")
        raise

    finally:
        pass
