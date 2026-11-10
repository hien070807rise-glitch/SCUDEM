"""
run_super_simulation.py - Kich ban chay kiem dinh sieu lon (Super Large Monte Carlo Verification).
Chay kiem dinh toan dien cac cot moc so lieu tu Giai doan 1 den Giai doan 5,
xuat bao cao JSON chi tiet kem sai so chuan (SE), khoang tin cay 95% va kiem dinh TOST.
"""

import sys
import os
import time
import json
import pickle
import numpy as np

# Thiet lap ma hoa dau ra an toan tren Windows
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)
from super_sim import run_simulation, stats_ci, paired_diff

CACHE_FILE = os.path.join(script_dir, "super_sim_cache.pkl")

def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)

def run_large_stage1_2(R=5000, seed=1, cache=None):
    if cache is not None and "stage1_2" in cache:
        print("[1/5] Giai doan 1 & 2 da co trong cache, tai lai...", flush=True)
        return cache["stage1_2"]
        
    print(f"\n[1/5] Chay kiem dinh Giai doan 1 & 2 (R={R}, seed={seed})...", flush=True)
    t0 = time.time()
    
    # Giai doan 1: Don doc
    r_e1_pass = run_simulation(dict(shared=False, Bc=0, Bs=0, b=0, kappa=1.0), R=R, seed=seed)
    r_e1_k = run_simulation(dict(shared=False, Bc=0, Bs=0, b=0, kappa=0.3), R=R, seed=seed)
    r_e1_bg0 = run_simulation(dict(shared=False, Bc=0, Bs=0, b=0, Bg=0), R=R, seed=seed)
    
    e1_pass_m, e1_pass_se, _, _ = stats_ci(r_e1_pass['food'].mean(0))
    e1_k_m, e1_k_se, _, _ = stats_ci(r_e1_k['food'].mean(0))
    e1_bg0_m, e1_bg0_se, _, _ = stats_ci(r_e1_bg0['food'].mean(0))
    
    # Giai doan 2: Hai ca the chung kho khong lien lac
    r_e0 = run_simulation(dict(shared=True, Bc=0, Bs=0, b=0, kappa=1.0), R=R, seed=seed)
    e0_m, e0_se, _, _ = stats_ci(r_e0['food'].mean(0))
    
    phi_0 = e0_m / e1_pass_m
    
    # Quet khoang cach d0
    d0_vals = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 140.0, 200.0]
    decomp = []
    for d0 in d0_vals:
        sp = run_simulation(dict(shared=True, Bc=0, Bs=0.2, d0=d0, b=0), R=R, seed=seed)
        fl = run_simulation(dict(shared=True, Bc=4.0, Bs=0.2, d0=d0, b=3), R=R, seed=seed)
        esp_m, esp_se, _, _ = stats_ci(sp['food'].mean(0))
        efl_m, efl_se, _, _ = stats_ci(fl['food'].mean(0))
        dc_m, dc_se, _, _ = paired_diff(fl['food'].mean(0), sp['food'].mean(0))
        decomp.append(dict(d0=d0, E_sp=esp_m, se_sp=esp_se, E_fl=efl_m, se_fl=efl_se, Delta_c=dc_m, se_dc=dc_se))
        
    print(f"  -> E1 passive: {e1_pass_m:.4f} +/- {e1_pass_se:.4f}")
    print(f"  -> E1 (kappa=0.3): {e1_k_m:.4f} +/- {e1_k_se:.4f}")
    print(f"  -> E1 (Bg=0): {e1_bg0_m:.4f} +/- {e1_bg0_se:.4f}")
    print(f"  -> E0 shared: {e0_m:.4f} +/- {e0_se:.4f} (E0/E1 = {phi_0:.4f})")
    print(f"  -> Hoan thanh Giai doan 1 & 2 trong {time.time()-t0:.1f}s")
    
    res = dict(
        E1_passive=dict(val=e1_pass_m, se=e1_pass_se),
        E1_kappa03=dict(val=e1_k_m, se=e1_k_se),
        E1_Bg0=dict(val=e1_bg0_m, se=e1_bg0_se),
        E0_shared=dict(val=e0_m, se=e0_se),
        E0_over_E1=phi_0,
        decomp=decomp,
    )
    if cache is not None:
        cache["stage1_2"] = res
        save_cache(cache)
    return res

def run_large_stage3(R=5000, seed=1, cache=None):
    if cache is not None and "stage3" in cache:
        print("[2/5] Giai doan 3 da co trong cache, tai lai...", flush=True)
        return cache["stage3"]
        
    print(f"\n[2/5] Chay kiem dinh Giai doan 3: Nhieu, Cong & Bac chi tiet (R={R})...", flush=True)
    t0 = time.time()
    
    # 1. Nhieu toa do sigma_c voi cong bat (gamma=1, Bc=8, Bs=0)
    sig_vals = [0.0, 30.0, 60.0, 100.0, 250.0]
    r_e0 = run_simulation(dict(shared=True, Bc=0, Bs=0, b=0), R=R, seed=seed)
    e0_arr = r_e0['food'].mean(0)
    
    noise_curve = []
    for s in sig_vals:
        res = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, sigc=s, gamma=1.0), R=R, seed=seed)
        val_m, val_se, _, _ = paired_diff(res['food'].mean(0), e0_arr)
        noise_curve.append(dict(sigc=s, dE_vs_E0=val_m, se=val_se))
        
    # Uoc luong sigma_1/2 noi suy
    v0 = noise_curve[0]['dE_vs_E0']
    half_v = v0 / 2.0
    sig_arr = np.array([pt['sigc'] for pt in noise_curve])
    val_arr = np.array([pt['dE_vs_E0'] for pt in noise_curve])
    sig_half_est = float(np.interp(half_v, val_arr[::-1], sig_arr[::-1]))
    
    # 2. Tat cong gamma=0 o nguoi nhan ngay tho
    gate_off_curve = []
    for s in [0.0, 60.0, 80.0, 100.0]:
        res_off = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, sigc=s, gate='off'), R=R, seed=seed)
        val_m, val_se, _, _ = paired_diff(res_off['food'].mean(0), e0_arr)
        gate_off_curve.append(dict(sigc=s, dE_vs_E0=val_m, se=val_se))
        
    # 3. Kiem chung b=2 vs b=3 trong 1D
    r_b2 = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=2, sigc=0.0), R=R, seed=seed)
    r_b3 = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, sigc=0.0), R=R, seed=seed)
    diff_b23_m, diff_b23_se, _, _ = paired_diff(r_b3['food'].mean(0), r_b2['food'].mean(0))
    
    # 4. Bac chi tiet toi uu theo chi phi c0
    costs = [0.0, 0.002, 0.004, 0.008, 0.016]
    b_opt_results = {}
    r_b_list = {b: run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=b, sigc=0.0), R=R, seed=seed) for b in [1, 2, 3, 4]}
    
    for c0 in costs:
        net_b = {}
        for b in [1, 2, 3, 4]:
            q_int = r_b_list[b]['qint'] # (2, R)
            cost_arr = (c0 * b * q_int) / 600.0
            net_val = r_b_list[b]['food'] - cost_arr
            d_val_m, _, _, _ = paired_diff(net_val.mean(0), e0_arr)
            net_b[b] = d_val_m
        opt_b = max(net_b.keys(), key=lambda k: net_b[k])
        b_opt_results[c0] = dict(net_values=net_b, optimal_b=opt_b)
        
    print(f"  -> Nhieu nua gia tri sigma_1/2 noi suy: {sig_half_est:.1f} m")
    print(f"  -> Hieu so cap b=3 - b=2: {diff_b23_m:+.5f} +/- {diff_b23_se:.5f}")
    print(f"  -> Bac toi uu: c0=0.004 -> b={b_opt_results[0.004]['optimal_b']}, c0=0.016 -> b={b_opt_results[0.016]['optimal_b']}")
    print(f"  -> Hoan thanh Giai doan 3 trong {time.time()-t0:.1f}s")
    
    res = dict(
        noise_curve=noise_curve,
        sigma_half_est=sig_half_est,
        gate_off_curve=gate_off_curve,
        diff_b23=dict(diff=diff_b23_m, se=diff_b23_se),
        b_opt_results=b_opt_results,
    )
    if cache is not None:
        cache["stage3"] = res
        save_cache(cache)
    return res

def run_large_stage4(R=5000, seed=1, cache=None):
    if cache is not None and "stage4" in cache:
        print("[3/5] Giai doan 4 da co trong cache, tai lai...", flush=True)
        return cache["stage4"]
        
    print(f"\n[3/5] Chay kiem dinh Giai doan 4: Lua doi, Huong & Canh tranh (R={R})...", flush=True)
    t0 = time.time()
    
    # 1. So sanh lua doi keo ve phia nguoi nhan vs nhieu cung do lon (Delta = 60m, t=0 vs sigc=60m)
    r_h = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, t=1.0), R=R, seed=seed)
    r_lie60 = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, t=0.0, Delta=60.0, lietype='default'), R=R, seed=seed)
    r_noise60 = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, sigc=60.0, noise_mask=(1.0, 0.0)), R=R, seed=seed)
    
    harm_B_lie_m, harm_B_lie_se, _, _ = paired_diff(r_lie60['food'][1], r_h['food'][1])
    gain_A_lie_m, gain_A_lie_se, _, _ = paired_diff(r_lie60['food'][0], r_h['food'][0])
    harm_B_noise_m, harm_B_noise_se, _, _ = paired_diff(r_noise60['food'][1], r_h['food'][1])
    
    harm_ratio = harm_B_lie_m / harm_B_noise_m if harm_B_noise_m != 0 else 0
    harm_gain_ratio = abs(harm_B_lie_m) / gain_A_lie_m if gain_A_lie_m != 0 else 0
    
    # 2. Ba huong lua doi: toward, rand, away
    r_rand = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, t=0.0, Delta=60.0, lietype='rand'), R=R, seed=seed)
    r_away = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, t=0.0, Delta=60.0, lietype='away'), R=R, seed=seed)
    harm_rand_m, harm_rand_se, _, _ = paired_diff(r_rand['food'][1], r_h['food'][1])
    harm_away_m, harm_away_se, _, _ = paired_diff(r_away['food'][1], r_h['food'][1])
    
    # 3. Quet he so canh tranh dung chung kho s in [0, 1]
    s_vals = [0.0, 0.25, 0.5, 0.67, 0.75, 1.0]
    comp_sweep = []
    for s_fr in s_vals:
        if s_fr == 0.0:
            h_s = run_simulation(dict(shared=False, Bc=8.0, Bs=0, b=3, t=1.0), R=R, seed=seed)
            l_s = run_simulation(dict(shared=False, Bc=8.0, Bs=0, b=3, t=0.0, Delta=60.0), R=R, seed=seed)
        elif s_fr == 1.0:
            h_s = r_h
            l_s = r_lie60
        else:
            h_s = run_simulation(dict(shared=False, couple=s_fr, Bc=8.0, Bs=0, b=3, t=1.0), R=R, seed=seed)
            l_s = run_simulation(dict(shared=False, couple=s_fr, Bc=8.0, Bs=0, b=3, t=0.0, Delta=60.0), R=R, seed=seed)
        dA_m, dA_se, _, _ = paired_diff(l_s['food'][0], h_s['food'][0])
        dB_m, dB_se, _, _ = paired_diff(l_s['food'][1], h_s['food'][1])
        comp_sweep.append(dict(s=s_fr, gain_A=dA_m, se_A=dA_se, harm_B=dB_m, se_B=dB_se))
        
    # 4. Ti le Hai / Loi khi co canh tranh can thiep (lam=1) va giu patch (kappa=0.3)
    r_h_lam = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, t=1.0, lam=1.0), R=R, seed=seed)
    r_l_lam = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, t=0.0, Delta=60.0, lam=1.0), R=R, seed=seed)
    dA_lam_m, _, _, _ = paired_diff(r_l_lam['food'][0], r_h_lam['food'][0])
    dB_lam_m, _, _, _ = paired_diff(r_l_lam['food'][1], r_h_lam['food'][1])
    ratio_lam = abs(dB_lam_m) / dA_lam_m if dA_lam_m != 0 else 0
    
    r_h_kap = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, t=1.0, kappa=0.3), R=R, seed=seed)
    r_l_kap = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, t=0.0, Delta=60.0, kappa=0.3), R=R, seed=seed)
    dA_kap_m, _, _, _ = paired_diff(r_l_kap['food'][0], r_h_kap['food'][0])
    dB_kap_m, _, _, _ = paired_diff(r_l_kap['food'][1], r_h_kap['food'][1])
    ratio_kap = abs(dB_kap_m) / dA_kap_m if dA_kap_m != 0 else 0
    
    # 5. Nguong bang-bang t0 cua nguoi nhan
    r_p0 = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, t=0.0, trust=0.0), R=R, seed=seed)
    bb_points = []
    for t_test in [0.2, 0.35, 0.37, 0.4, 0.5]:
        r_t = run_simulation(dict(shared=True, Bc=8.0, Bs=0, b=3, t=t_test, Delta=60.0, trust=1.0), R=R, seed=seed)
        net_diff_m, _, _, _ = paired_diff(r_t['food'][1], r_p0['food'][1])
        bb_points.append(dict(t=t_test, EB_minus_Ebo=net_diff_m))
        
    print(f"  -> Thiet hai B (lua doi Delta=60m, t=0): {harm_B_lie_m:.4f} +/- {harm_B_lie_se:.4f}")
    print(f"  -> Thiet hai B (nhieu sigc=60m): {harm_B_noise_m:.4f} +/- {harm_B_noise_se:.4f}")
    print(f"  -> Ti so Hai (Lua doi / Nhieu): {harm_ratio:.2f} lan (can tren)")
    print(f"  -> Ti le Hai B / Loi A (co so): {harm_gain_ratio:.2f} : 1")
    print(f"  -> Ti le Hai B / Loi A (lam=1): {ratio_lam:.2f} : 1; (kappa=0.3): {ratio_kap:.2f} : 1")
    print(f"  -> Huong ngau nhien: {harm_rand_m:.4f} ({harm_rand_m/harm_B_noise_m:.2f}x nhieu); Day ra xa: {harm_away_m:.4f}")
    print(f"  -> Hoan thanh Giai doan 4 trong {time.time()-t0:.1f}s")
    
    res = dict(
        harm_B_lie=dict(val=harm_B_lie_m, se=harm_B_lie_se),
        gain_A_lie=dict(val=gain_A_lie_m, se=gain_A_lie_se),
        harm_B_noise=dict(val=harm_B_noise_m, se=harm_B_noise_se),
        harm_ratio_lie_noise=harm_ratio,
        harm_gain_ratio_base=harm_gain_ratio,
        harm_gain_ratio_lam=ratio_lam,
        harm_gain_ratio_kap=ratio_kap,
        harm_rand=dict(val=harm_rand_m, se=harm_rand_se),
        harm_away=dict(val=harm_away_m, se=harm_away_se),
        comp_sweep=comp_sweep,
        bang_bang_points=bb_points,
    )
    if cache is not None:
        cache["stage4"] = res
        save_cache(cache)
    return res

def run_large_stage5_and_tost(R=5000, seed=1, cache=None):
    if cache is not None and "stage5" in cache:
        print("[4/5] Giai doan 5 da co trong cache, tai lai...", flush=True)
        return cache["stage5"]
        
    print(f"\n[4/5] Chay kiem dinh Giai doan 5 & TOST Surrogate (R={R})...", flush=True)
    t0 = time.time()
    
    # 1. Base point Stage 5: M0, V0
    base_p = dict(shared=True, Bs=0.2, Bc=8.0, b=3, c0=0.004, d0=50.0, Rc=50.0, t=1.0, gamma=1.0)
    ref_p = dict(shared=True, Bs=0.2, Bc=0.0, b=0, c0=0.004, d0=50.0, Rc=50.0)
    
    r_base = run_simulation(base_p, R=R, seed=seed)
    r_ref = run_simulation(ref_p, R=R, seed=seed)
    
    M0_m, M0_se, _, _ = stats_ci(r_base['net_E'].mean(0))
    Mref_m, Mref_se, _, _ = stats_ci(r_ref['net_E'].mean(0))
    V0_m, V0_se, _, _ = paired_diff(r_base['net_E'].mean(0), r_ref['net_E'].mean(0))
    
    # 2. Tornado swings: d0 in [20, 140] vs sigc in [0, 100]
    r_d0_20 = run_simulation({**base_p, 'd0': 20.0}, R=R, seed=seed)
    r_d0_140 = run_simulation({**base_p, 'd0': 140.0}, R=R, seed=seed)
    r_ref_20 = run_simulation({**ref_p, 'd0': 20.0}, R=R, seed=seed)
    r_ref_140 = run_simulation({**ref_p, 'd0': 140.0}, R=R, seed=seed)
    V_d0_20_m, _, _, _ = paired_diff(r_d0_20['net_E'].mean(0), r_ref_20['net_E'].mean(0))
    V_d0_140_m, _, _, _ = paired_diff(r_d0_140['net_E'].mean(0), r_ref_140['net_E'].mean(0))
    delta_V_d0 = V_d0_20_m - V_d0_140_m
    
    r_sig100 = run_simulation({**base_p, 'sigc': 100.0}, R=R, seed=seed)
    V_sig100_m, _, _, _ = paired_diff(r_sig100['net_E'].mean(0), r_ref['net_E'].mean(0))
    delta_V_sigc = V0_m - V_sig100_m
    
    # 3. Tuong tac Id0_t va Igam_t
    r_d0_100 = run_simulation({**base_p, 'd0': 100.0}, R=R, seed=seed)
    r_t0 = run_simulation({**base_p, 't': 0.0, 'Delta': 60.0}, R=R, seed=seed)
    r_d0_t0 = run_simulation({**base_p, 'd0': 100.0, 't': 0.0, 'Delta': 60.0}, R=R, seed=seed)
    r_ref_100 = run_simulation({**ref_p, 'd0': 100.0}, R=R, seed=seed)
    
    v0_arr = r_base['net_E'].mean(0) - r_ref['net_E'].mean(0)
    va_arr = r_d0_100['net_E'].mean(0) - r_ref_100['net_E'].mean(0)
    vb_arr = r_t0['net_E'].mean(0) - r_ref['net_E'].mean(0)
    vab_arr = r_d0_t0['net_E'].mean(0) - r_ref_100['net_E'].mean(0)
    
    I_d0_t = vab_arr - va_arr - vb_arr + v0_arr
    I_d0_t_m, I_d0_t_se, _, _ = stats_ci(I_d0_t)
    
    r_gam05 = run_simulation({**base_p, 'gamma': 0.5}, R=R, seed=seed)
    r_gam_t0 = run_simulation({**base_p, 'gamma': 0.5, 't': 0.0, 'Delta': 60.0}, R=R, seed=seed)
    va_g = r_gam05['net_E'].mean(0) - r_ref['net_E'].mean(0)
    vab_g = r_gam_t0['net_E'].mean(0) - r_ref['net_E'].mean(0)
    I_gam_t = vab_g - va_g - vb_arr + v0_arr
    I_gam_t_m, I_gam_t_se, _, _ = stats_ci(I_gam_t)
    
    # 4. Kiem dinh tuong duong TOST cho mo hinh rut gon (Option C)
    print("  -> Thuc hien kiem dinh TOST tren mo hinh Surrogate...")
    r_full = run_simulation(base_p, R=R, seed=seed)
    r_surr = run_simulation({**base_p, 'surrogate': True, 'alpha_eff': 1.0}, R=R, seed=seed)
    
    diff_pair = r_full['net_E'].mean(0) - r_surr['net_E'].mean(0)
    diff_m, diff_se, ci_low, ci_high = stats_ci(diff_pair)
    
    delta_eq = 0.002
    tost_pass = (ci_low >= -delta_eq) and (ci_high <= delta_eq)
    
    print(f"  -> M0: {M0_m:.4f} +/- {M0_se:.4f}, V0: {V0_m:.4f} +/- {V0_se:.4f}")
    print(f"  -> Tornado swing: Delta V (d0): {delta_V_d0:.4f}, Delta V (sigc): {delta_V_sigc:.4f} (gap ~{delta_V_d0/delta_V_sigc:.1f}x)")
    print(f"  -> Tuong tac: I(d0, t) = {I_d0_t_m:+.4f} +/- {I_d0_t_se:.4f} ({I_d0_t_m/I_d0_t_se:.1f} SE)")
    print(f"  -> Tuong tac: I(gamma, t) = {I_gam_t_m:+.4f} +/- {I_gam_t_se:.4f} ({I_gam_t_m/I_gam_t_se:.1f} SE)")
    print(f"  -> TOST Surrogate: mean diff = {diff_m:+.5f} +/- {diff_se:.5f}, 95% CI: [{ci_low:+.5f}; {ci_high:+.5f}]")
    print(f"     Nguong tuong duong: [{-delta_eq:.3f}; {delta_eq:.3f}] -> Ket luan TOST: {'DAT (EQUIVALENT)' if tost_pass else 'CHUA DAT'}")
    print(f"  -> Hoan thanh Giai doan 5 trong {time.time()-t0:.1f}s")
    
    res = dict(
        M0=dict(val=M0_m, se=M0_se),
        V0=dict(val=V0_m, se=V0_se),
        delta_V_d0=delta_V_d0,
        delta_V_sigc=delta_V_sigc,
        swing_ratio=delta_V_d0 / delta_V_sigc if delta_V_sigc > 0 else 0,
        I_d0_t=dict(val=I_d0_t_m, se=I_d0_t_se, z_score=I_d0_t_m/I_d0_t_se),
        I_gam_t=dict(val=I_gam_t_m, se=I_gam_t_se, z_score=I_gam_t_m/I_gam_t_se),
        TOST_surrogate=dict(
            diff_mean=diff_m,
            diff_se=diff_se,
            ci_95=[ci_low, ci_high],
            delta_eq=delta_eq,
            tost_passed=bool(tost_pass),
        ),
    )
    if cache is not None:
        cache["stage5"] = res
        save_cache(cache)
    return res

def main():
    print("====================================================================")
    print("BAT DAU CHAY BO MO PHONG SIEU LON (SUPER-LARGE MONTE CARLO AUDIT)")
    print("====================================================================")
    t_start = time.time()
    
    cache = load_cache()
    R_SUPER = 5000
    
    res1_2 = run_large_stage1_2(R=R_SUPER, seed=10, cache=cache)
    res3 = run_large_stage3(R=R_SUPER, seed=20, cache=cache)
    res4 = run_large_stage4(R=R_SUPER, seed=30, cache=cache)
    res5 = run_large_stage5_and_tost(R=R_SUPER, seed=40, cache=cache)
    
    total_time = time.time() - t_start
    print("====================================================================")
    print(f"TOAN BO MO PHONG SIEU LON HOAN TAT TRONG {total_time:.1f}s ({total_time/60.0:.2f} phut)!")
    print("====================================================================")
    
    master_report = dict(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        R_replicates=R_SUPER,
        total_runtime_seconds=total_time,
        stage1_2=res1_2,
        stage3=res3,
        stage4=res4,
        stage5=res5,
    )
    
    out_json = os.path.join(script_dir, "super_simulation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(master_report, f, indent=2, ensure_ascii=False)
    print(f"Da xuat ket qua day du ra file: {out_json}")

if __name__ == '__main__':
    main()
