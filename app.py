import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

plt.rcParams["axes.unicode_minus"] = False
G = 9.81

st.set_page_config(
    page_title="5조 복합 배관계 해석 및 손실 저감 비교 프로그램",
    layout="wide"
)

# ==========================================
# 0. 사용자가 원하는 기본 초기값 설정 구역
# ==========================================
DEFAULTS = {
    "q_m3h": 5.0,
    "p1_kpa": 150.0,
    "p2_kpa": 0.0,
    "dz_m": 2.0,
    "rho": 1000.0,
    "mu": 0.001,
    "length_m": 30.0,
    "od_mm": 33.4,
    "t_mm": 3.38,
    "eps_mm": 0.045,
    "n_el": 2,
    "k_el": 0.75,
    "n_val": 1,
    "k_val": 5.0,
    "mode": "배관 교체 허용",
    "dmin_mm": 20.0,
    "dmax_mm": 80.0,
    "custom_d_mm": 40.0,
    "vmax_ms": 3.0,
    "hpmax_m": 20.0,
    "new_n_el": 2,
    "new_k_el": 0.75,
    "new_n_val": 1,
    "new_k_val": 5.0,
}

# session_state 초기화
for k, v in DEFAULTS.items():
    if f"inp_{k}" not in st.session_state:
        st.session_state[f"inp_{k}"] = v


def reset_defaults_state():
    for k, v in DEFAULTS.items():
        st.session_state[f"inp_{k}"] = v


# ==========================================
# 1. Darcy 마찰계수
# ==========================================
def friction_factor(re, relative_roughness, method="sj"):
    if re <= 0:
        return 0.0

    if re < 2000:
        return 64.0 / re

    if re <= 4000:
        return np.nan

    if method == "sj":
        return 0.25 / (
            np.log10(
                relative_roughness / 3.7
                + 5.74 / re**0.9
            ) ** 2
        )

    f = 0.02

    for _ in range(200):
        next_f = (
            -2.0 * np.log10(
                relative_roughness / 3.7
                + 2.51 / (re * np.sqrt(f))
            )
        ) ** -2

        if abs(next_f - f) < 1e-12:
            return next_f

        f = next_f

    raise RuntimeError("Colebrook 반복계산 수렴 실패")


# ==========================================
# 2. 배관계 계산
# ==========================================
def calculate_system(
    q_m3h, p1_kpa, d_m, l_m, eps_mm,
    rho, mu, z_diff, n_el, k_el, n_val, k_val,
    p2_target_kpa=0.0, method="sj"
):
    q = q_m3h / 3600.0
    area = np.pi * d_m**2 / 4.0
    velocity = q / area
    re = rho * velocity * d_m / mu
    relative_roughness = eps_mm / 1000.0 / d_m

    f = friction_factor(re, relative_roughness, method)
    velocity_head = velocity**2 / (2.0 * G)

    major_loss = f * (l_m / d_m) * velocity_head
    elbow_loss = n_el * k_el * velocity_head
    valve_loss = n_val * k_val * velocity_head
    minor_loss = elbow_loss + valve_loss
    total_loss = major_loss + minor_loss

    p2_no_pump = (
        p1_kpa
        - rho * G * (z_diff + total_loss) / 1000.0
    )

    required_p1 = (
        p2_target_kpa
        + rho * G * (z_diff + total_loss) / 1000.0
    )

    balance_head = (
        (p2_target_kpa - p1_kpa) * 1000.0 / (rho * G)
        + z_diff + total_loss
    )

    additional_hp = (
        max(0.0, balance_head)
        if np.isfinite(balance_head) else np.nan
    )

    if np.isfinite(total_loss) and total_loss > 0:
        major_ratio = major_loss / total_loss * 100.0
        minor_ratio = minor_loss / total_loss * 100.0
    else:
        major_ratio = 0.0
        minor_ratio = 0.0

    if re == 0:
        regime = "정지"
    elif re < 2000:
        regime = "층류"
    elif re <= 4000:
        regime = "전이영역"
    else:
        regime = "난류"

    return {
        "Q_m3h": q_m3h,
        "D_mm": d_m * 1000.0,
        "Velocity": velocity,
        "Reynolds": re,
        "Regime": regime,
        "Friction_Factor": f,
        "Major_Loss": major_loss,
        "Elbow_Loss": elbow_loss,
        "Valve_Loss": valve_loss,
        "Minor_Loss": minor_loss,
        "Total_Loss": total_loss,
        "P2_kPa": p2_no_pump,
        "Required_P1_kPa": required_p1,
        "Additional_Hp_m": additional_hp,
        "Major_Ratio": major_ratio,
        "Minor_Ratio": minor_ratio,
        "Transition": 2000 <= re <= 4000
    }


def calc_reduction(base_loss, target_loss):
    if (
        np.isfinite(base_loss)
        and base_loss > 0
        and np.isfinite(target_loss)
    ):
        diff = base_loss - target_loss
        pct = diff / base_loss * 100.0
        return diff, pct
    return np.nan, np.nan


# ==========================================
# 3. 헤더 및 사이드바 입력 위젯
# ==========================================
st.title("5조 복합 배관계 해석 및 손실 저감 비교 프로그램")
st.caption(
    "원하는 값을 직접 입력하면 표와 6분할 그래프, 실제 배관 계통도가 즉시 갱신됩니다. | "
    "압력 기준: 게이지압 | 마찰계수: Darcy 기준 | 전이영역: 2000 ≤ Re ≤ 4000 (손실계산 및 선정 제외)"
)

with st.sidebar:
    st.button("기본값으로 되돌리기", on_click=reset_defaults_state, use_container_width=True)
    st.header("기본 입력 조건")

    c1, c2 = st.columns(2)
    with c1:
        q_val = st.number_input("목표유량(m³/h)", key="inp_q_m3h", step=0.5)
        p2_val = st.number_input("목표 출구압력(kPa,g)", key="inp_p2_kpa", step=5.0)
        rho_val = st.number_input("밀도(kg/m³)", key="inp_rho", step=10.0)
        length_val = st.number_input("직관 총길이(m)", key="inp_length_m", step=1.0)
        od_val = st.number_input("기존 외경 OD(mm)", key="inp_od_mm", step=0.1)
        elbow_val = int(st.number_input("기존 90° 엘보 개수", key="inp_n_el", step=1))
        valve_val = int(st.number_input("기존 밸브 개수", key="inp_n_val", step=1))
    with c2:
        p1_val = st.number_input("공급압력(kPa,g)", key="inp_p1_kpa", step=5.0)
        dz_val = st.number_input("고도차(z2-z1,m)", key="inp_dz_m", step=0.5)
        mu_val = st.number_input("점도(Pa·s)", key="inp_mu", format="%.6f")
        eps_val = st.number_input("배관조도(mm)", key="inp_eps_mm", format="%.4f")
        t_val = st.number_input("기존 벽두께 t(mm)", key="inp_t_mm", step=0.05)
        kel_val = st.number_input("기존 엘보 K값", key="inp_k_el", step=0.05)
        kval_val = st.number_input("기존 밸브 K값", key="inp_k_val", step=0.5)

    st.divider()
    st.subheader("관경 탐색, 제한조건 및 Fitting 개선 비교")
    mode_val = st.selectbox("설계 조건", ["배관 교체 허용", "기존 배관 유지"], key="inp_mode")
    replacement = mode_val == "배관 교체 허용"

    c3, c4 = st.columns(2)
    with c3:
        dmin_val = st.number_input("탐색 최소 내경(mm)", key="inp_dmin_mm", step=1.0)
        custom_d_val = st.number_input("비교 내경(mm)", key="inp_custom_d_mm", step=1.0)
        hpmax_val = st.number_input("추가 펌프수두 한도(m)", key="inp_hpmax_m", step=1.0)
        new_elbow_val = int(st.number_input("개선안 엘보 개수", key="inp_new_n_el", step=1))
        new_valve_val = int(st.number_input("개선안 밸브 개수", key="inp_new_n_val", step=1))
    with c4:
        dmax_val = st.number_input("탐색 최대 내경(mm)", key="inp_dmax_mm", step=1.0)
        vmax_val = st.number_input("허용 최대유속(m/s)", key="inp_vmax_ms", step=0.1)
        st.empty()
        new_kel_val = st.number_input("개선안 엘보 K값", key="inp_new_k_el", step=0.05)
        new_kval_val = st.number_input("개선안 밸브 K값", key="inp_new_k_val", step=0.5)


# ==========================================
# 4. 입력 검증 및 계산
# ==========================================
numeric_vals = [
    q_val, p1_val, p2_val, dz_val, rho_val, mu_val,
    length_val, od_val, t_val, eps_val, elbow_val, kel_val,
    valve_val, kval_val, dmin_val, dmax_val, custom_d_val,
    vmax_val, hpmax_val, new_elbow_val, new_kel_val,
    new_valve_val, new_kval_val
]

if not all(np.isfinite(v) for v in numeric_vals):
    st.error("입력 오류: 유한하지 않은 숫자가 포함되어 있습니다.")
    st.stop()

current_d_mm = od_val - 2.0 * t_val

if (
    q_val <= 0 or rho_val <= 0 or mu_val <= 0
    or od_val <= 0 or t_val < 0 or current_d_mm <= 0
):
    st.error("입력 오류: 유량·밀도·점도·외경·내경은 양수, 벽두께는 0 이상이어야 합니다.")
    st.stop()

nonnegative = [
    length_val, eps_val,
    elbow_val, kel_val, valve_val, kval_val,
    new_elbow_val, new_kel_val, new_valve_val, new_kval_val
]
if any(v < 0 for v in nonnegative):
    st.error("입력 오류: 길이·조도·부속 개수·K값에는 음수를 입력할 수 없습니다.")
    st.stop()

if vmax_val <= 0 or hpmax_val < 0:
    st.error("입력 오류: 최대유속은 양수, 추가 펌프수두 한도는 0 이상이어야 합니다.")
    st.stop()

if not (0 < dmin_val < dmax_val):
    st.error("입력 오류: 탐색 범위는 0 < 최소 내경 < 최대 내경이어야 합니다.")
    st.stop()


def run_case(d_mm, q=None, length=None, n_el=None, kel=None, n_val=None, kval=None, method="sj"):
    return calculate_system(
        q_m3h=q_val if q is None else q,
        p1_kpa=p1_val,
        d_m=d_mm / 1000.0,
        l_m=length_val if length is None else length,
        eps_mm=eps_val,
        rho=rho_val,
        mu=mu_val,
        z_diff=dz_val,
        n_el=elbow_val if n_el is None else n_el,
        k_el=kel_val if kel is None else kel,
        n_val=valve_val if n_val is None else n_val,
        k_val=kval_val if kval is None else kval,
        p2_target_kpa=p2_val,
        method=method
    )


current = run_case(current_d_mm)
custom_d_valid = custom_d_val > 0
custom_res = run_case(custom_d_val) if custom_d_valid else None

scan_diameters = np.linspace(dmin_val, dmax_val, 301)
scan_df = pd.DataFrame([run_case(d) for d in scan_diameters])
scan_df["Feasible"] = (
    (~scan_df["Transition"])
    & np.isfinite(scan_df["Total_Loss"])
    & (scan_df["Velocity"] <= vmax_val)
    & (scan_df["Additional_Hp_m"] <= hpmax_val)
)

feasible = scan_df[scan_df["Feasible"]]
min_feasible = feasible.iloc[0] if not feasible.empty else None
low_loss_alt = feasible.iloc[-1] if not feasible.empty else None

fitting_changed = (
    new_elbow_val != elbow_val
    or not np.isclose(new_kel_val, kel_val)
    or new_valve_val != valve_val
    or not np.isclose(new_kval_val, kval_val)
)

fitting_alt = run_case(
    current_d_mm,
    n_el=new_elbow_val,
    kel=new_kel_val,
    n_val=new_valve_val,
    kval=new_kval_val
)


# ==========================================
# 5. 기존 배관 해석 및 손실 저감 대안 요약
# ==========================================
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("[기존 배관 해석]")
    st.write(f"- **관 내경:** {current_d_mm:.3f} mm")
    st.write(f"- **평균 유속:** {current['Velocity']:.3f} m/s")
    st.write(f"- **Reynolds 수:** {current['Reynolds']:,.0f}")
    st.write(f"- **유동 상태:** {current['Regime']}")

    if current["Transition"]:
        st.warning("손실수두·압력 계산: 전이영역(2000 ≤ Re ≤ 4000)으로 제외")
    else:
        st.write(f"- **Darcy 마찰계수:** {current['Friction_Factor']:.6f}")
        st.write(f"- **직관 손실수두:** {current['Major_Loss']:.3f} m")
        st.write(f"- **엘보 손실수두:** {current['Elbow_Loss']:.3f} m")
        st.write(f"- **밸브 손실수두:** {current['Valve_Loss']:.3f} m")
        st.write(f"- **총 손실수두:** {current['Total_Loss']:.3f} m")
        st.write(f"- **주손실 / 부손실:** {current['Major_Ratio']:.1f}% / {current['Minor_Ratio']:.1f}%")
        st.write(f"- **추가 펌프 없는 출구압력:** {current['P2_kPa']:.3f} kPa,g")
        st.write(f"- **목표 출구압력 기준 필요 입구압력:** {current['Required_P1_kPa']:.3f} kPa,g")
        st.write(f"- **추가 펌프 필요수두:** {current['Additional_Hp_m']:.3f} m")

with col_right:
    st.subheader("[관경 탐색 및 손실 저감 대안 비교]")
    spacing = (dmax_val - dmin_val) / 300.0
    st.write(f"- **설계 조건:** {mode_val}")
    st.write(f"- **탐색 내경:** {dmin_val:.3f} ~ {dmax_val:.3f} mm (후보 {len(scan_df)}개, 간격 {spacing:.3f} mm)")
    st.write(f"- **제한조건:** 유속 ≤ {vmax_val:.3f} m/s, 추가 펌프수두 ≤ {hpmax_val:.3f} m")

    if feasible.empty:
        st.warning("선정 결과: 제한조건을 만족하는 탐색 후보 없음")
    else:
        st.success(f"① 최소 적합 내경: {min_feasible['D_mm']:.3f} mm (유속 {min_feasible['Velocity']:.3f} m/s, 총 손실 {min_feasible['Total_Loss']:.3f} m)")
        st.info(f"② 저손실 대안(큰 내경): {low_loss_alt['D_mm']:.3f} mm (유속 {low_loss_alt['Velocity']:.3f} m/s, 총 손실 {low_loss_alt['Total_Loss']:.3f} m)")
        diff_min, pct_min = calc_reduction(current["Total_Loss"], min_feasible["Total_Loss"])
        diff_low, pct_low = calc_reduction(current["Total_Loss"], low_loss_alt["Total_Loss"])
        if np.isfinite(diff_low):
            st.write(f"- **기존 대비 손실 변화:** 최소 적합 내경 {diff_min:+.3f} m ({pct_min:+.1f}%) | 저손실 대안 {diff_low:+.3f} m ({pct_low:+.1f}%)")

    if custom_d_valid:
        if custom_res["Transition"] or not np.isfinite(custom_res["Total_Loss"]):
            st.write(f"- **사용자 지정 비교 내경({custom_d_val:.3f} mm):** 전이영역으로 손실 비교 제외")
        else:
            c_diff, c_pct = calc_reduction(current["Total_Loss"], custom_res["Total_Loss"])
            st.write(f"- **사용자 지정 비교 내경({custom_d_val:.3f} mm):** 총 손실 {custom_res['Total_Loss']:.3f} m ({c_diff:+.3f} m, {c_pct:+.1f}%)")

    if fitting_changed:
        if current["Transition"] or fitting_alt["Transition"]:
            st.write("- **기존 배관 + Fitting 변경안:** 전이영역 포함으로 손실 비교 제외")
        else:
            f_diff, f_pct = calc_reduction(current["Total_Loss"], fitting_alt["Total_Loss"])
            st.write(f"- **기존 배관 + Fitting 변경안:** 총 손실 {fitting_alt['Total_Loss']:.3f} m ({f_diff:+.3f} m, {f_pct:+.1f}%)")


def make_summary_row(label, res, is_base=False):
    if is_base:
        diff, pct = 0.0, 0.0
    else:
        diff, pct = calc_reduction(current["Total_Loss"], res["Total_Loss"])
    return {
        "구분": label,
        "내경(mm)": round(float(res["D_mm"]), 3),
        "유속(m/s)": round(float(res["Velocity"]), 3),
        "Re": round(float(res["Reynolds"]), 0),
        "총 손실수두(m)": round(float(res["Total_Loss"]), 3) if np.isfinite(res["Total_Loss"]) else np.nan,
        "추가 펌프수두(m)": round(float(res["Additional_Hp_m"]), 3) if np.isfinite(res["Additional_Hp_m"]) else np.nan,
        "기존 대비 손실감소(m)": round(float(diff), 3) if np.isfinite(diff) else np.nan,
        "손실 감소율(%)": round(float(pct), 1) if np.isfinite(pct) else np.nan,
    }


summary_rows = [make_summary_row("기존 배관", current, is_base=True)]
if custom_d_valid:
    summary_rows.append(make_summary_row("사용자 지정 내경", custom_res))
if min_feasible is not None:
    summary_rows.append(make_summary_row("최소 적합 내경", min_feasible))
    summary_rows.append(make_summary_row("저손실 대안(큰 내경)", low_loss_alt))
if fitting_changed:
    summary_rows.append(make_summary_row("기존관+Fitting변경", fitting_alt))

st.subheader("[주요 대안 비교표]")
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

st.subheader("[탐색 후보 요약: 최대 5개]")
columns = ["D_mm", "Velocity", "Reynolds", "Total_Loss", "Additional_Hp_m", "Feasible"]
table = (feasible.head(5) if not feasible.empty else scan_df.head(5))[columns].rename(columns={
    "D_mm": "내경(mm)",
    "Velocity": "유속(m/s)",
    "Reynolds": "Re",
    "Total_Loss": "총 손실수두(m)",
    "Additional_Hp_m": "추가 펌프수두(m)",
    "Feasible": "조건 충족"
})
st.dataframe(table.round(3), use_container_width=True)

check_d = (
    float(low_loss_alt["D_mm"])
    if (replacement and low_loss_alt is not None)
    else (float(min_feasible["D_mm"]) if min_feasible is not None else current_d_mm)
)
check = run_case(check_d)

with st.expander(f"[계산 검증: 내경 {check_d:.3f} mm]"):
    if check["Transition"]:
        st.write("검증 결과: 전이영역으로 계산 제외")
    else:
        reference = run_case(check_d, method="colebrook")
        actual = check["Total_Loss"]
        expected = reference["Total_Loss"]
        st.write(f"- 프로그램 총 손실수두: {actual:.6f} m")
        st.write(f"- 기준계산 총 손실수두: {expected:.6f} m")
        if expected > 0:
            error = abs(actual - expected) / abs(expected) * 100.0
            st.write(f"- 기준계산 대비 상대차: {error:.4f}%")
        else:
            st.write(f"- 기준계산 대비 절대차: {abs(actual - expected):.6f} m")
        st.write("- 비교 기준: Swamee–Jain 근사식 / Colebrook 반복계산 (계산법 간 비교, 실험값 비교 아님)")
        k_zero = run_case(check_d, kel=0.0, kval=0.0)
        l_zero = run_case(check_d, length=0.0)
        st.write(f"- K=0 검증 절대차: {abs(k_zero['Total_Loss'] - check['Major_Loss']):.3e} m")
        st.write(f"- L=0 검증 절대차: {abs(l_zero['Total_Loss'] - check['Minor_Loss']):.3e} m")


# ==========================================
# 6. 기존 vs 개선안 비교 6분할 그래프
# ==========================================
flow_max = max(20.0, 1.2 * q_val)
flows = np.linspace(0.0, flow_max, 41)
flow_curr_df = pd.DataFrame([run_case(current_d_mm, q=q) for q in flows])
flow_low_df = pd.DataFrame([run_case(float(low_loss_alt["D_mm"]), q=q) for q in flows]) if low_loss_alt is not None else None
flow_fit_df = (
    pd.DataFrame([
        run_case(current_d_mm, q=q, n_el=new_elbow_val, kel=new_kel_val, n_val=new_valve_val, kval=new_kval_val)
        for q in flows
    ])
    if fitting_changed else None
)

fig, axes = plt.subplots(2, 3, figsize=(17, 9))

ax = axes[0, 0]
ax.plot(flows, flow_curr_df["Total_Loss"], "k-", linewidth=2, label=f"Existing ({current_d_mm:.1f} mm)")
if flow_low_df is not None:
    ax.plot(flows, flow_low_df["Total_Loss"], color="teal", linestyle="--", linewidth=2, label=f"Low-loss ({low_loss_alt['D_mm']:.1f} mm)")
if flow_fit_df is not None:
    ax.plot(flows, flow_fit_df["Total_Loss"], color="darkorange", linestyle="-.", linewidth=2, label="Existing D + New Fitting")
ax.axvline(q_val, color="gray", linestyle=":", label=f"Target Q ({q_val:.1f})")
ax.set_title("Flow vs Total Head Loss (Before vs After)")
ax.set_xlabel("Flow rate (m³/h)")
ax.set_ylabel("Total head loss (m)")
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[0, 1]
ax.plot(flows, flow_curr_df["Reynolds"], color="green", linewidth=2, label=f"Existing ({current_d_mm:.1f} mm)")
if flow_low_df is not None:
    ax.plot(flows, flow_low_df["Reynolds"], color="teal", linestyle="--", linewidth=2, label=f"Low-loss ({low_loss_alt['D_mm']:.1f} mm)")
ax.axhspan(2000, 4000, color="orange", alpha=0.25, label="Transition")
ax.axvline(q_val, color="gray", linestyle=":", label=f"Target Q ({q_val:.1f})")
ax.set_title("Flow vs Reynolds (Before vs After)")
ax.set_xlabel("Flow rate (m³/h)")
ax.set_ylabel("Reynolds number")
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[0, 2]
bar_labels = ["Existing"]
major_vals = [current["Major_Loss"] if np.isfinite(current["Major_Loss"]) else 0.0]
minor_vals = [current["Minor_Loss"] if np.isfinite(current["Minor_Loss"]) else 0.0]

if custom_d_valid:
    bar_labels.append(f"Custom\n({custom_res['D_mm']:.1f}mm)")
    major_vals.append(custom_res["Major_Loss"] if np.isfinite(custom_res["Major_Loss"]) else 0.0)
    minor_vals.append(custom_res["Minor_Loss"] if np.isfinite(custom_res["Minor_Loss"]) else 0.0)
if min_feasible is not None:
    bar_labels.append(f"Min D\n({min_feasible['D_mm']:.1f}mm)")
    major_vals.append(min_feasible["Major_Loss"])
    minor_vals.append(min_feasible["Minor_Loss"])
if low_loss_alt is not None:
    bar_labels.append(f"Low-loss\n({low_loss_alt['D_mm']:.1f}mm)")
    major_vals.append(low_loss_alt["Major_Loss"])
    minor_vals.append(low_loss_alt["Minor_Loss"])
if fitting_changed:
    bar_labels.append("New\nFitting")
    major_vals.append(fitting_alt["Major_Loss"] if np.isfinite(fitting_alt["Major_Loss"]) else 0.0)
    minor_vals.append(fitting_alt["Minor_Loss"] if np.isfinite(fitting_alt["Minor_Loss"]) else 0.0)

x_pos = np.arange(len(bar_labels))
ax.bar(x_pos, major_vals, label="Major loss", color="steelblue")
ax.bar(x_pos, minor_vals, bottom=major_vals, label="Minor loss", color="coral")
for idx, (maj, mino) in enumerate(zip(major_vals, minor_vals)):
    tot = maj + mino
    ax.text(idx, tot, f"{tot:.2f}m", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=8)
ax.set_title("Head Loss Breakdown by Case")
ax.set_ylabel("Head loss (m)")
ax.legend(fontsize=8)
ax.grid(True, axis="y")

ax = axes[1, 0]
ax.plot(scan_df["D_mm"], scan_df["Velocity"], "b.-", label="Velocity")
ax.axhline(vmax_val, color="r", linestyle="--", label="Velocity limit")
ax.axvline(current_d_mm, color="purple", linestyle="-.", label="Existing D")
if custom_d_valid:
    ax.axvline(custom_d_val, color="gold", linestyle="-", label="Custom D")
if min_feasible is not None:
    ax.axvline(min_feasible["D_mm"], color="green", linestyle=":", label="Min feasible D")
if low_loss_alt is not None:
    ax.axvline(low_loss_alt["D_mm"], color="teal", linestyle="--", alpha=0.8, label="Low-loss D")
ax.set_title("Diameter vs Velocity")
ax.set_xlabel("Internal diameter (mm)")
ax.set_ylabel("Velocity (m/s)")
ax.legend(fontsize=8)
ax.grid(True)

ax = axes[1, 1]
ax.plot(scan_df["D_mm"], scan_df["Total_Loss"], "b-", label="Total loss (m)")
ax.plot(scan_df["D_mm"], scan_df["Additional_Hp_m"], "k.-", label="Add. pump head (m)")
ax.axhline(hpmax_val, color="r", linestyle="--", label="Pump head limit")
ax.axvline(current_d_mm, color="purple", linestyle="-.", label="Existing D")
if custom_d_valid:
    ax.axvline(custom_d_val, color="gold", linestyle="-", label="Custom D")
if min_feasible is not None:
    ax.axvline(min_feasible["D_mm"], color="green", linestyle=":", label="Min feasible D")
if low_loss_alt is not None:
    ax.axvline(low_loss_alt["D_mm"], color="teal", linestyle="--", alpha=0.8, label="Low-loss D")
ax.set_title("Diameter vs Total Loss & Pump Head")
ax.set_xlabel("Internal diameter (mm)")
ax.set_ylabel("Head (m)")
ax.legend(fontsize=8)
ax.grid(True)

k_values = np.linspace(0.0, max(10.0, 2.0 * max(kval_val, new_kval_val)), 25)
k_curr = [run_case(current_d_mm, kval=k) for k in k_values]
ax = axes[1, 2]
ax.plot(k_values, [r["Total_Loss"] for r in k_curr], "k-", label="Existing (Total loss)")
ax.plot(k_values, [r["Minor_Loss"] for r in k_curr], "b--", label="Existing (Minor loss)")
if fitting_changed:
    k_new = [
        run_case(current_d_mm, n_el=new_elbow_val, kel=new_kel_val, n_val=new_valve_val, kval=k)
        for k in k_values
    ]
    ax.plot(k_values, [r["Total_Loss"] for r in k_new], color="darkorange", linestyle="-.", label="New Fitting (Total loss)")
ax.set_title("Valve K vs Loss (Before vs After)")
ax.set_xlabel("K per valve")
ax.set_ylabel("Head loss (m)")
ax.legend(fontsize=8)
ax.grid(True)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# ==========================================
# 7. 실제 배관 계통도 (P&ID 스타일 입상/직각 배관 구조)
# ==========================================
def draw_valve_symbol(ax, x, y, horizontal=True, label=None):
    s = 0.22
    if horizontal:
        tri1 = plt.Polygon([[x - s, y - s * 0.7], [x - s, y + s * 0.7], [x, y]], closed=True, facecolor="#f59e0b", edgecolor="#1e293b", lw=1.3, zorder=6)
        tri2 = plt.Polygon([[x + s, y - s * 0.7], [x + s, y + s * 0.7], [x, y]], closed=True, facecolor="#f59e0b", edgecolor="#1e293b", lw=1.3, zorder=6)
        ax.add_patch(tri1)
        ax.add_patch(tri2)
        ax.plot([x, x], [y, y + s * 1.3], color="#1e293b", lw=1.5, zorder=6)
        ax.plot([x - s * 0.7, x + s * 0.7], [y + s * 1.3, y + s * 1.3], color="#1e293b", lw=1.8, zorder=6)
        if label:
            ax.text(x, y - s * 1.25, label, ha="center", va="top", fontsize=8, fontweight="bold", color="#b45309")
    else:
        tri1 = plt.Polygon([[x - s * 0.7, y - s], [x + s * 0.7, y - s], [x, y]], closed=True, facecolor="#f59e0b", edgecolor="#1e293b", lw=1.3, zorder=6)
        tri2 = plt.Polygon([[x - s * 0.7, y + s], [x + s * 0.7, y + s], [x, y]], closed=True, facecolor="#f59e0b", edgecolor="#1e293b", lw=1.3, zorder=6)
        ax.add_patch(tri1)
        ax.add_patch(tri2)
        ax.plot([x, x + s * 1.3], [y, y], color="#1e293b", lw=1.5, zorder=6)
        ax.plot([x + s * 1.3, x + s * 1.3], [y - s * 0.7, y + s * 0.7], color="#1e293b", lw=1.8, zorder=6)
        if label:
            ax.text(x - s * 1.2, y, label, ha="right", va="center", fontsize=8, fontweight="bold", color="#b45309")


def draw_centrifugal_pump(ax, cx, cy, active=True, add_hp=0.0):
    r = 0.42
    color = "#dc2626" if active else "#64748b"
    fill = "#fef2f2" if active else "#f8fafc"
    base = plt.Polygon(
        [[cx - r * 0.85, cy - r * 1.35], [cx + r * 0.85, cy - r * 1.35],
         [cx + r * 0.55, cy - r * 0.75], [cx - r * 0.55, cy - r * 0.75]],
        closed=True, facecolor=color, edgecolor="#1e293b", lw=1.2, zorder=4
    )
    ax.add_patch(base)
    circle = plt.Circle((cx, cy), r, facecolor=fill, edgecolor=color, lw=2.5, zorder=5)
    ax.add_patch(circle)
    ax.plot([cx, cx + r * 1.35], [cy + r, cy + r], color=color, lw=3.2, zorder=4)
    ax.text(cx, cy, "P-101", ha="center", va="center", fontsize=8.5, fontweight="bold", color=color, zorder=6)
    status = f"PUMP ON (+{add_hp:.2f} m)" if active else "BYPASS / NO EXTRA PUMP"
    ax.text(cx, cy - r * 1.55, status, ha="center", va="top", fontsize=8.5, fontweight="bold", color=color)


def draw_tank(ax, x, y, width=1.1, height=1.5, title="TANK", info=""):
    rect = plt.Rectangle((x - width / 2, y - height / 2), width, height, facecolor="#f1f5f9", edgecolor="#334155", lw=2, zorder=3)
    ax.add_patch(rect)
    fluid_h = height * 0.65
    fluid = plt.Rectangle((x - width / 2 + 0.03, y - height / 2 + 0.03), width - 0.06, fluid_h, facecolor="#bae6fd", edgecolor="none", zorder=2)
    ax.add_patch(fluid)
    wx = np.linspace(x - width / 2 + 0.05, x + width / 2 - 0.05, 30)
    wy = (y - height / 2 + fluid_h) + 0.04 * np.sin(np.linspace(0, 3 * np.pi, 30))
    ax.plot(wx, wy, color="#0284c7", lw=1.5, zorder=4)
    ax.text(x, y + height / 2 + 0.15, title, ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0f172a")
    if info:
        ax.text(x, y - height / 2 - 0.18, info, ha="center", va="top", fontsize=8.5, color="#334155")


def plot_real_piping_diagram(current_d, opt_d, is_replacement, dz, l_m, n_el, n_val, add_hp, p1, p2, q_m3h, case_label):
    fig, ax = plt.subplots(figsize=(14, 5.6))

    display_d = opt_d if (is_replacement and opt_d is not None) else current_d
    pipe_lw = np.clip(display_d / 10.0, 2.8, 8.5)
    pipe_color = "#0284c7" if is_replacement else "#1e293b"
    pump_active = bool(np.isfinite(add_hp) and add_hp > 1e-6)

    y_base = 1.8
    visual_dz = np.clip(dz * 0.35, -1.0, 2.2)
    if abs(dz) > 0 and abs(visual_dz) < 0.7:
        visual_dz = 0.7 if dz > 0 else -0.7
    y_top = y_base + visual_dz

    draw_tank(ax, 1.0, y_base + 0.25, title="SUCTION TANK", info=f"P1 = {p1:.1f} kPa,g\nz1 = 0.0 m")
    ax.plot([1.55, 2.7], [y_base, y_base], color="#334155", lw=pipe_lw, zorder=3)
    draw_centrifugal_pump(ax, 3.1, y_base, active=pump_active, add_hp=add_hp)

    y_disch = y_base + 0.42
    x_start = 3.65
    x_end = 10.6

    pts = [(x_start, y_disch)]
    if n_el <= 0:
        pts.append((x_end, y_top))
    elif n_el == 1:
        pts.append((x_end, y_disch))
        pts.append((x_end, y_top))
    elif n_el == 2:
        x_mid = 6.6
        pts.append((x_mid, y_disch))
        pts.append((x_mid, y_top))
        pts.append((x_end, y_top))
    else:
        n_steps = n_el // 2
        xs = np.linspace(x_start + 1.0, x_end - 1.0, n_steps + 1)
        ys = np.linspace(y_disch, y_top, n_steps + 1)
        for i in range(n_steps):
            pts.append((xs[i], ys[i]))
            pts.append((xs[i], ys[i + 1]))
        if n_el % 2 == 1:
            pts.append((x_end, ys[-1]))
            pts.append((x_end, y_top + 0.4))
        else:
            pts.append((x_end, y_top))

    px = [p[0] for p in pts]
    py = [p[1] for p in pts]
    ax.plot(px, py, color="#0f172a", lw=pipe_lw + 2.2, solid_joinstyle="miter", zorder=3)
    ax.plot(px, py, color=pipe_color, lw=pipe_lw, solid_joinstyle="miter", zorder=4)

    for idx, (ex, ey) in enumerate(pts[1:-1], start=1):
        ax.scatter([ex], [ey], s=55, color="#10b981", edgecolor="#0f172a", zorder=6)
        offset_y = 0.22 if idx % 2 == 1 else -0.28
        ax.text(ex, ey + offset_y, f"EL-{idx}", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#047857")

    if n_val > 0:
        horiz_segs = []
        for i in range(len(pts) - 1):
            x1, y1_s = pts[i]
            x2, y2_s = pts[i + 1]
            if abs(y1_s - y2_s) < 1e-6 and abs(x2 - x1) > 0.6:
                horiz_segs.append((x1, x2, y1_s))
        if horiz_segs:
            for v_idx in range(n_val):
                seg = horiz_segs[v_idx % len(horiz_segs)]
                frac = (v_idx // len(horiz_segs) + 1) / (n_val // len(horiz_segs) + 2)
                vx = seg[0] + frac * (seg[1] - seg[0])
                vy = seg[2]
                draw_valve_symbol(ax, vx, vy, horizontal=True, label=f"V-{v_idx + 1}")

    last_x, last_y = pts[-1]
    ax.plot([last_x, 11.45], [last_y, last_y], color="#0f172a", lw=pipe_lw + 2.2, zorder=3)
    ax.plot([last_x, 11.45], [last_y, last_y], color=pipe_color, lw=pipe_lw, zorder=4)
    draw_tank(ax, 12.0, last_y + 0.25, title="DISCHARGE TANK", info=f"Target P2 = {p2:.1f} kPa,g\nz2 = {dz:+.1f} m")

    ax.annotate("", xy=(2.45, y_base + 0.25), xytext=(1.75, y_base + 0.25),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#0284c7"))
    ax.text(2.1, y_base + 0.35, f"Q={q_m3h:.2f} m³/h", ha="center", va="bottom", fontsize=8, color="#0284c7", fontweight="bold")

    ax.plot([10.9, 10.9], [y_base, last_y], color="#64748b", linestyle="--", lw=1.2)
    ax.annotate("", xy=(10.9, last_y), xytext=(10.9, y_base),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color="#475569"))
    ax.text(11.0, (y_base + last_y) / 2, f"Δz = {dz:+.2f} m", ha="left", va="center", fontsize=8.5, fontweight="bold", color="#334155")

    spec_title = f"[{case_label}] PIPE ID: {display_d:.2f} mm (Existing: {current_d:.2f} mm)" if is_replacement else f"[{case_label}] PIPE ID: {display_d:.2f} mm"
    spec_detail = f"Total Straight Length L = {l_m:.1f} m  |  90° Elbows: {n_el} ea  |  Valves: {n_val} ea"
    ax.text(
        6.5, 0.45, f"{spec_title}\n{spec_detail}",
        ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor=pipe_color, lw=1.8)
    )

    ax.set_xlim(0.1, 13.0)
    ax.set_ylim(-0.2, max(4.8, last_y + 1.5))
    ax.axis("off")
    plt.tight_layout()
    return fig


st.subheader("[최적 배관 계통도 (실제 배관 배치 구조)]")
if low_loss_alt is not None:
    opt_d_val = float(low_loss_alt["D_mm"])
    add_hp_val = float(low_loss_alt["Additional_Hp_m"])
    case_lbl = "LOW-LOSS ALTERNATIVE"
elif min_feasible is not None:
    opt_d_val = float(min_feasible["D_mm"])
    add_hp_val = float(min_feasible["Additional_Hp_m"])
    case_lbl = "MIN FEASIBLE"
else:
    opt_d_val = None
    add_hp_val = float(current["Additional_Hp_m"])
    case_lbl = "EXISTING PIPE"

fig_sch = plot_real_piping_diagram(
    current_d=current_d_mm,
    opt_d=opt_d_val,
    is_replacement=replacement,
    dz=dz_val,
    l_m=length_val,
    n_el=new_elbow_val if fitting_changed else elbow_val,
    n_val=new_valve_val if fitting_changed else valve_val,
    add_hp=add_hp_val,
    p1=p1_val,
    p2=p2_val,
    q_m3h=q_val,
    case_label=case_lbl
)
st.pyplot(fig_sch)
plt.close(fig_sch)
