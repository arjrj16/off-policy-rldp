"""
Empirical test: does the pretrained IDQL critic's ranking of candidate action
chunks carry useful signal, or is it noise?

Loads the EXACT init of the harmful arm (BC actor state_1300 + pretrained critic
state_250) and runs two tests:

PART A (offline, val.npz):
  - Q-spread across N candidates at a state vs the critic's held-out Bellman RMSE
    (signal-to-noise of the ranking).
  - Do the two Q-heads (Q1, Q2) agree on which candidate is best? (is the ranking
    internally robust, or init-dependent noise)
  - Spearman( Q(s,a_data), discounted MC chunk return-to-go ) (does Q track real
    outcomes at all).

PART B (online, decisive): roll out the BC actor for full episodes under three
rules that select from the SAME candidate pool each step:
  - argmax-Q  (what eval does -> produced the -1000 curve)
  - random    (ignore the critic = BC policy)
  - min-Q     (anti-select)
If the ranking helps:  argmax > random > min.  If noise: all equal.
If actively harmful:   argmax < random.
"""

import os, sys, math, time
import numpy as np
import torch
from omegaconf import OmegaConf
import hydra

OmegaConf.register_new_resolver("eval", eval, replace=True)
OmegaConf.register_new_resolver("round_up", math.ceil, replace=True)
OmegaConf.register_new_resolver("round_down", math.floor, replace=True)
OmegaConf.register_new_resolver("now", lambda *a: "ranktest", replace=True)

REPO = "/home/melwani/67920/code2/off-policy-rldp-ogbench"
CFG = f"{REPO}/cfg/ogbench/finetune/cube-double/ft_idql_diffusion_mlp.yaml"
DEVICE = "cuda:0"
N = 10            # candidates per step (matches eval_sample_num)
torch.manual_seed(0); np.random.seed(0)

def log(*a):
    print(*a, flush=True)

from model.diffusion.diffusion_rwr import RWRDiffusion  # IDQL samples via this (super().forward)

def spearman(x, y):
    # x,y: 1d tensors
    rx = x.argsort().argsort().float(); ry = y.argsort().argsort().float()
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx*ry).sum() / (rx.norm()*ry.norm() + 1e-9))

# ----------------------------------------------------------------------------
log("="*80); log("Loading config + model (BC actor state_1300 + critic state_250)")
cfg = OmegaConf.load(CFG)
model = hydra.utils.instantiate(cfg.model).to(DEVICE)
model.eval()
log(f"  critic_path = {cfg.model.critic_path}")
log(f"  base_policy = {cfg.model.network_path}")

@torch.no_grad()
def sample_and_score(cond_state, n, deterministic=True):
    """cond_state: (B,T,D). Returns samples (n,B,H,A), q/q1/q2 (n,B)."""
    B, T, D = cond_state.shape
    rep = cond_state.unsqueeze(0).repeat(n, 1, 1, 1).reshape(n*B, T, D)
    samples = RWRDiffusion.forward(model, {"state": rep}, deterministic=deterministic)
    q1, q2 = model.target_q({"state": rep}, samples)
    H, A = samples.shape[1:]
    return (samples.reshape(n, B, H, A),
            torch.min(q1, q2).reshape(n, B), q1.reshape(n, B), q2.reshape(n, B))

# ============================ PART A: offline ===============================
log("="*80); log("PART A  (offline, held-out val.npz)")
try:
    from agent.dataset.sequence import StitchedChunkQLearningDataset
    val_path = f"{os.environ['DPPO_DATA_DIR']}/ogbench/cube-double/val.npz"
    ds = StitchedChunkQLearningDataset(dataset_path=val_path, horizon_steps=cfg.horizon_steps,
                                       cond_steps=cfg.cond_steps, device=DEVICE)
    nB = min(512, len(ds))
    idxs = np.random.choice(len(ds), nB, replace=False)
    S  = torch.stack([ds[i].conditions["state"] for i in idxs]).to(DEVICE)       # (nB,T,D)
    NS = torch.stack([ds[i].conditions["next_state"] for i in idxs]).to(DEVICE)
    A  = torch.stack([ds[i].actions for i in idxs]).to(DEVICE)                    # (nB,H,Ad)
    R  = torch.stack([ds[i].rewards for i in idxs]).to(DEVICE).view(-1)
    Dn = torch.stack([ds[i].dones for i in idxs]).to(DEVICE).view(-1)

    with torch.no_grad():
        # noise floor: held-out Bellman RMSE of Q(s,a_data) vs r + gamma V(s')(1-done)
        q1d, q2d = model.target_q({"state": S}, A)
        vnext = model.critic_v({"state": NS}).view(-1)
        tgt = R + cfg.train.gamma * vnext * (1.0 - Dn)
        rmse = float(torch.sqrt(((torch.min(q1d, q2d) - tgt)**2).mean()))
        meanQ = float(torch.min(q1d, q2d).mean())

        # candidate spread + Q1/Q2 agreement (sample N per state from the actor)
        samp, q, q1, q2 = sample_and_score(S, N, deterministic=True)            # (N,nB)
        spread = (q.max(0).values - q.min(0).values)                            # (nB,)
        agree_argmax = float((q1.argmax(0) == q2.argmax(0)).float().mean())
        sp = np.mean([spearman(q1[:, b], q2[:, b]) for b in range(nB)])

    log(f"  mean |Q| (val, data actions)      : {abs(meanQ):8.1f}")
    log(f"  held-out Bellman RMSE (noise floor): {rmse:8.2f}")
    log(f"  mean Q-spread across {N} candidates : {float(spread.mean()):8.2f}   (median {float(spread.median()):.2f})")
    log(f"  ==> SNR = spread / RMSE            : {float(spread.mean())/rmse:8.3f}   (>~1 = usable, <~1 = in the noise)")
    log(f"  Q1 vs Q2 agree on BEST candidate  : {agree_argmax*100:7.1f}%   (random chance = {100.0/N:.0f}%)")
    log(f"  Q1 vs Q2 mean Spearman over cands : {sp:8.3f}   (1=identical ranking, 0=unrelated)")

    # does Q track real outcomes? Spearman(Q(s,a_data), MC chunk return-to-go)
    try:
        gamma = float(cfg.train.gamma); H = cfg.horizon_steps
        raw = np.load(val_path); rew = raw["rewards"].astype(np.float64); tl = raw["traj_lengths"]
        masks = raw["masks"] if "masks" in raw else (rew > -0.5)
        succ = (masks < 0.5) if "masks" in raw else masks
        # chunk return-to-go per chunk-start index, within each trajectory, early-stop at success
        rtg = np.zeros(len(rew)); start = 0
        for L in tl:
            seg_r = rew[start:start+L]; seg_s = succ[start:start+L]
            # chunk reward at each t (sum of H steps, early stop at success)
            cr = np.zeros(L)
            for t in range(L):
                e = min(t+H, L); s_idx = np.where(seg_s[t:e])[0]
                cr[t] = seg_r[t:t+(s_idx[0]+1)].sum() if len(s_idx) else seg_r[t:e].sum()
            # discounted return-to-go over chunks stepping by H
            g = np.zeros(L)
            for t in range(L-1, -1, -1):
                nxt = g[t+H] if t+H < L else 0.0
                g[t] = cr[t] + gamma*nxt
                if len(np.where(seg_s[t:min(t+H,L)])[0]):  # terminal chunk: no bootstrap
                    g[t] = cr[t]
            rtg[start:start+L] = g; start += L
        # map our sampled val indices back to absolute step indices
        abs_idx = np.array([ds.indices[i][0] for i in idxs])
        mc = torch.tensor(rtg[abs_idx], dtype=torch.float32, device=DEVICE)
        sp_mc = spearman(torch.min(q1d, q2d), mc)
        log(f"  Spearman( Q(s,a_data), MC return ): {sp_mc:8.3f}   (does Q rank real outcomes)")
    except Exception as e:
        log(f"  [MC-corr skipped: {e}]")
except Exception as e:
    import traceback; log("PART A failed:", e); traceback.print_exc()

# ============================ PART B: online ================================
log("="*80); log("PART B  (online rollouts, paired by reset seed)")
try:
    from env.gym_utils import make_async
    act_steps = cfg.act_steps
    venv = make_async(cfg.env.name, env_type="ogbench", num_envs=cfg.env.n_envs,
                      asynchronous=True, max_episode_steps=cfg.env.max_episode_steps,
                      wrappers=cfg.env.wrappers, render=False, render_offscreen=False,
                      obs_dim=cfg.obs_dim, action_dim=cfg.action_dim)
    n_envs = cfg.env.n_envs; T = cfg.env.max_episode_steps  # 500

    @torch.no_grad()
    def rollout(mode):
        venv.seed([100 + i for i in range(n_envs)])   # identical resets across modes
        obs = venv.reset_arg(options_list=[{} for _ in range(n_envs)])
        if isinstance(obs, list):
            obs = {k: np.stack([o[k] for o in obs]) for k in obs[0].keys()}
        ret = np.zeros(n_envs); best = np.full(n_envs, -1e9); succ = np.zeros(n_envs, bool)
        done_any = np.zeros(n_envs, bool)
        steps = T // act_steps + 2
        for st in range(steps):
            if done_any.all():
                break
            cond = torch.from_numpy(obs["state"]).float().to(DEVICE)
            samp, q, _, _ = sample_and_score(cond, N, deterministic=True)       # (N,n_envs,..)
            if mode == "argmax":   idx = q.argmax(0)
            elif mode == "min":    idx = q.argmin(0)
            elif mode == "random": idx = torch.randint(0, N, (n_envs,), device=DEVICE)
            sel = samp[idx, torch.arange(n_envs, device=DEVICE)]                # (n_envs,H,A)
            act = sel[:, :act_steps].cpu().numpy()
            obs, rew, term, trunc, info = venv.step(act)
            # count reward only until an env's first episode end (avoid mixing resets)
            live = ~done_any
            ret[live] += rew[live]; best = np.maximum(best, np.where(live, rew, -1e9))
            for i in range(n_envs):
                if live[i] and "success" in info[i] and bool(np.asarray(info[i]["success"]).reshape(-1)[-1]):
                    succ[i] = True
            done_any |= (term | trunc)
        return ret.mean(), (best/act_steps).mean(), succ.mean(), int((~done_any).sum())

    log(f"  {n_envs} envs, full {T}-step episodes, N={N} candidates/step\n")
    log(f"  {'mode':>8} | {'mean episode reward':>20} | {'mean best/step':>14} | {'success':>8}")
    log("  " + "-"*60)
    res = {}
    for mode in ["random", "argmax", "min"]:
        t0 = time.time()
        r, b, s, nlive = rollout(mode)
        res[mode] = r
        log(f"  {mode:>8} | {r:>20.2f} | {b:>14.4f} | {s:>7.1%}    ({time.time()-t0:.0f}s, {nlive} unfinished)")
    log("")
    log(f"  argmax - random = {res['argmax']-res['random']:+.1f}  (>0 critic helps, <0 critic HURTS vs BC)")
    log(f"  random - min    = {res['random']-res['min']:+.1f}  (>0 means ranking carries SOME signal)")
    venv.close()
except Exception as e:
    import traceback; log("PART B failed:", e); traceback.print_exc()

log("="*80); log("DONE")
