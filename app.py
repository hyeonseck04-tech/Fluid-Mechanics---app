import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

plt.rcParams["axes.unicode_minus"] = False
G = 9.81

st.set_page_config(page_title="5조 복합 배관계 해석 프로그램", layout="wide")


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


# ==========================================
# 3. 사이드바 입력 위젯 (Streamlit)
# ==========================================
st.title("5조 복합 배관계 해석 프로그램")
st.caption(
    "압력 기준: 게이지압 | 입구·출구: 동일 내경의 관 단면 | "
    "마찰계수: Darcy 기준 | 전이영역: 2000 ≤ Re ≤ 4000 (손실계산 및 선정 제외)"
)

with st.sidebar:
    st.header("입력 조건")
    c1, c2 = st.columns(2)
    with c1:
        q_val = st.number_input("목표유량(m³/h)", value=5.0, step=0.5)
        p2_val = st.number_input("목표 출구압력(kPa,g)", value=0.0, step=5.0)
        rho_val = st.number_input("밀도(kg/m³)", value=1000.0, step=10.0)
        length_val = st.number_input("직관 총길이(m)", value=30.0, step=1.0)
        od_val = st.number_input("기존 외경 OD(mm)", value=33.4, step=0.1)
        elbow_val = int(st.number_input("90° 엘보 개수", value=2, step=1))
        valve_val = int(st.number_input("밸브 개수", value=1, step=1))
    with c2:
        p1_val = st.number_input("공급압력(kPa,g)", value=150.0, step=5.0)
        dz_val = st.number_input("고도차(z2-z1,m)", value=2.0, step=0.5)
        mu_val = st.number_input("점도(Pa·s)", value=0.001, format="%.6f")
        eps_val = st.number_input("배관조도(mm)", value=0.045, format="%.4f")
        t_val = st.number_input("기존 벽두께 t(mm)", value=3.38, step=0.05)
        kel_val = st.number_input("엘보 K값", value=0.75, step=0.05)
        kval_val = st.number_input("밸브 K값", value=5.0, step=0.5)

    st.divider()
    st.subheader("관경 탐색 및 제한조건")
    mode_val = st.selectbox("설계 조건", ["기존 배관 유지", "배관 교체 허용"])
    replacement = mode_val == "배관 교체 허용"

    c3, c4 = st.columns(2)
    with c3:
        dmin_val = st.number_input(
            "탐색 최소 내경(mm)", value=20.0, step=1.0, disabled=not replacement
        )
        vmax_val = st.number_input("허용 최대유속(m/s)", value=3.0, step=0.1)
    with c4:
        dmax_val = st.number_input(
            "탐색 최대 내경(mm)", value=80.0, step=1.0, disabled=not replacement
        )
        hpmax_val = st.number_input("추가 펌프수두 한도(m)", value=20.0, step=1.0)


# ==========================================
# 4. 입력 검증 및 계산
# ==========================================
numeric_vals = [
    q_val, p1_val, p2_val, dz_val, rho_val, mu_val,
    length_val, od_val, t_val, eps_val, elbow_val, kel_val,
    valve_val, kval_val, dmin_val, dmax_val, vmax_val, hpmax_val
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

if any(v < 0 for v in [length_val, eps_val, elbow_val, kel_val, valve_val, kval_val]):
    st.error("입력 오류: 길이·조도·부속 개수·K값에는 음수를 입력할 수 없습니다.")
    st.stop()

if vmax_val <= 0 or hpmax_val < 0:
    st.error("입력 오류: 최대유속은 양수, 추가 펌프수두 한도는 0 이상이어야 합니다.")
    st.stop()

if replacement and not (0 < dmin_val < dmax_val):
    st.error("입력 오류: 탐색 범위는 0 < 최소 내경 < 최대 내경이어야 합니다.")
    st.stop()


def run_case(d_mm, q=None, length=None, kel=None, kval=None, method="sj"):
    return calculate_system(
        q_m3h=q_val if q is None else q,
        p1_kpa=p1_val,
        d_m=d_mm / 1000.0,
        l_m=length_val if length is None else length,
        eps_mm=eps_val,
        rho=rho_val,
        mu=mu_val,
        z_diff=dz_val,
        n_el=elbow_val,
        k_el=kel_val if kel is None else kel,
        n_val=valve_val,
        k_val=kval_val if kval is None else kval,
        p2_target_kpa=p2_val,
        method=method
    )


current = run_case(current_d_mm)

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
        st.write(
            f"- **주손실 / 부손실:** "
            f"{current['Major_Ratio']:.1f}% / {current['Minor_Ratio']:.1f}%"
        )
        st.write(f"- **추가 펌프 없는 출구압력:** {current['P2_kPa']:.3f} kPa,g")
        st.write(f"- **목표 출구압력 기준 필요 입구압력:** {current['Required_P1_kPa']:.3f} kPa,g")
        st.write(f"- **추가 펌프 필요수두:** {current['Additional_Hp_m']:.3f} m")

diameters = (
    np.linspace(dmin_val, dmax_val, 301)
    if replacement else np.array([current_d_mm])
)

candidates = pd.DataFrame([run_case(d) for d in diameters])
candidates["Feasible"] = (
    (~candidates["Transition"])
    & np.isfinite(candidates["Total_Loss"])
    & (candidates["Velocity"] <= vmax_val)
    & (candidates["Additional_Hp_m"] <= hpmax_val)
)

feasible = candidates[candidates["Feasible"]]
selected = None

with col_right:
    st.subheader("[관경 선정]")
    st.write(f"- **설계 조건:** {mode_val}")
    st.write(f"- **허용 최대유속:** {vmax_val:.3f} m/s")
    st.write(f"- **추가 펌프수두 한도:** {hpmax_val:.3f} m")

    if replacement:
        spacing = (dmax_val - dmin_val) / 300.0
        st.write(f"- **탐색 내경:** {dmin_val:.3f} ~ {dmax_val:.3f} mm (후보 {len(candidates)}개, 간격 {spacing:.3f} mm)")

    if feasible.empty:
        st.warning("선정 결과: 제한조건을 만족하는 후보 없음")
    else:
        selected = feasible.iloc[0]
        if replacement:
            st.success(f"최소 적합 탐색 내경: {selected['D_mm']:.3f} mm")
        else:
            st.success("선정 결과: 기존 배관 제한조건 충족")

        st.write(f"- **선정 유속:** {selected['Velocity']:.3f} m/s")
        st.write(f"- **선정 총 손실수두:** {selected['Total_Loss']:.3f} m")
        st.write(f"- **선정 추가 펌프 필요수두:** {selected['Additional_Hp_m']:.3f} m")

        if selected["Additional_Hp_m"] == 0:
            st.caption("압력 조건: 기존 공급압력으로 목표 출구압력 이상 확보 가능 (목표유량 고정 기준)")

st.subheader("[후보 계산표: 최대 5개]")
columns = ["D_mm", "Velocity", "Reynolds", "Total_Loss", "Additional_Hp_m", "Feasible"]
table = (feasible.head(5) if not feasible.empty else candidates.head(5))[columns].rename(columns={
    "D_mm": "내경(mm)",
    "Velocity": "유속(m/s)",
    "Reynolds": "Re",
    "Total_Loss": "총 손실수두(m)",
    "Additional_Hp_m": "추가 펌프수두(m)",
    "Feasible": "조건 충족"
})
st.dataframe(table.round(3), use_container_width=True)

check_d = float(selected["D_mm"]) if selected is not None else current_d_mm
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

        if check["Reynolds"] > 4000:
            st.write("- 비교 기준: Swamee–Jain 근사식 / Colebrook 반복계산 (계산법 간 비교, 실험값 비교 아님)")
        else:
            st.write("- 비교 기준: 층류 Darcy 마찰계수 64/Re")

        k_zero = run_case(check_d, kel=0.0, kval=0.0)
        l_zero = run_case(check_d, length=0.0)
        st.write(f"- K=0 검증 절대차: {abs(k_zero['Total_Loss'] - check['Major_Loss']):.3e} m")
        st.write(f"- L=0 검증 절대차: {abs(l_zero['Total_Loss'] - check['Minor_Loss']):.3e} m")


# ==========================================
# 5. 그래프 출력 (st.pyplot)
# ==========================================
flow_max = max(20.0, 1.2 * q_val)
flows = np.linspace(0.0, flow_max, 41)
flow_df = pd.DataFrame([run_case(current_d_mm, q=q) for q in flows])

fig, axes = plt.subplots(2, 3, figsize=(17, 9))

# 1. 유량-손실수두
ax = axes[0, 0]
ax.plot(flows, flow_df["Total_Loss"], "k-", label="Total")
ax.plot(flows, flow_df["Major_Loss"], "b--", label="Major")
ax.plot(flows, flow_df["Minor_Loss"], "r:", label="Minor")
ax.set_title("Existing Pipe: Flow vs Head Loss")
ax.set_xlabel("Flow rate (m³/h)")
ax.set_ylabel("Head loss (m)")
ax.legend()
ax.grid(True)

# 2. 유량-Reynolds 수
ax = axes[0, 1]
ax.plot(flows, flow_df["Reynolds"], color="green")
ax.axhspan(2000, 4000, color="orange", alpha=0.25, label="Transition")
ax.set_title("Existing Pipe: Flow vs Reynolds")
ax.set_xlabel("Flow rate (m³/h)")
ax.set_ylabel("Reynolds number")
ax.legend()
ax.grid(True)

# 3. 손실 비율
ax = axes[0, 2]
if np.isfinite(current["Total_Loss"]) and current["Total_Loss"] > 0:
    ax.pie(
        [current["Major_Loss"], current["Minor_Loss"]],
        labels=["Major", "Minor"],
        autopct="%1.1f%%",
        startangle=90
    )
else:
    ax.text(0.5, 0.5, "No valid positive loss", ha="center", va="center", transform=ax.transAxes)
ax.set_title("Existing Pipe: Loss Ratio")

# 4. 관경-유속
ax = axes[1, 0]
ax.plot(candidates["D_mm"], candidates["Velocity"], "b.-")
ax.axhline(vmax_val, color="r", linestyle="--", label="Velocity limit")
ax.set_title("Diameter vs Velocity")
ax.set_xlabel("Internal diameter (mm)")
ax.set_ylabel("Velocity (m/s)")
ax.legend()
ax.grid(True)

# 5. 관경-추가 펌프수두
ax = axes[1, 1]
ax.plot(candidates["D_mm"], candidates["Additional_Hp_m"], "k.-")
ax.axhline(hpmax_val, color="r", linestyle="--", label="Pump head limit")
ax.set_title("Diameter vs Additional Pump Head")
ax.set_xlabel("Internal diameter (mm)")
ax.set_ylabel("Additional pump head (m)")
ax.legend()
ax.grid(True)

if selected is not None:
    for a in [axes[1, 0], axes[1, 1]]:
        a.axvline(selected["D_mm"], color="green", linestyle=":")

# 6. 밸브 K-손실수두
k_values = np.linspace(0.0, max(10.0, 2.0 * kval_val), 25)
k_results = [run_case(current_d_mm, kval=k) for k in k_values]

ax = axes[1, 2]
ax.plot(k_values, [r["Total_Loss"] for r in k_results], label="Total loss")
ax.plot(k_values, [r["Minor_Loss"] for r in k_results], "--", label="Minor loss")
ax.set_title("Existing Pipe: Valve K vs Loss")
ax.set_xlabel("K per valve (valve count fixed)")
ax.set_ylabel("Head loss (m)")
ax.legend()
ax.grid(True)

plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

# ==========================================
# 6. 최적 배관 계통 모식도 (Schematic)
# ==========================================
def plot_piping_schematic(current_d, opt_d, is_replacement, dz, l_m, n_el, n_val, add_hp, p1, p2):
    fig, ax = plt.subplots(figsize=(13, 4.2))

    # 좌표 설정 (좌 -> 우 흐름)
    x_inlet = 1.0
    x_pump = 2.5
    x_pipe_start = 4.0
    x_pipe_end = 9.0
    x_outlet = 10.2

    y1 = 2.0
    y2 = 2.0 + np.clip(dz * 0.25, -1.2, 1.5)  # 시각적 고도차 반영

    display_d = opt_d if (is_replacement and opt_d is not None) else current_d
    line_lw = np.clip(display_d / 12.0, 2.0, 8.0)

    # 입구 -> 펌프 -> 배관 시작 연결선
    ax.plot([x_inlet, x_pump], [y1, y1], color="#334155", lw=3)
    ax.plot([x_pump, x_pipe_start], [y1, y1], color="#334155", lw=3)

    # 펌프 심볼 (추가 펌프 수두가 0보다 클 때 강조)
    pump_color = "#dc2626" if (np.isfinite(add_hp) and add_hp > 1e-6) else "#64748b"
    pump_circle = plt.Circle((x_pump, y1), 0.35, facecolor="white", edgecolor=pump_color, lw=2.5, zorder=4)
    ax.add_patch(pump_circle)
    ax.text(x_pump, y1, "P", ha="center", va="center", fontsize=11, fontweight="bold", color=pump_color, zorder=5)
    pump_label = f"Pump (+{add_hp:.2f} m)" if (np.isfinite(add_hp) and add_hp > 1e-6) else "No Extra Pump"
    ax.text(x_pump, y1 - 0.65, pump_label, ha="center", va="top", fontsize=9, color=pump_color)

    # 메인 배관 (고도차 반영)
    pipe_color = "#0284c7" if is_replacement else "#0f172a"
    ax.plot([x_pipe_start, x_pipe_end], [y1, y2], color=pipe_color, lw=line_lw, solid_capstyle="round", zorder=3)

    # 밸브 마커 (배관 35% 지점)
    if n_val > 0:
        t_val = 0.35
        vx = x_pipe_start + t_val * (x_pipe_end - x_pipe_start)
        vy = y1 + t_val * (y2 - y1)
        ax.scatter([vx], [vy], marker="D", s=90, color="#f59e0b", edgecolor="black", zorder=5)
        ax.text(vx, vy + 0.35, f"Valve x{n_val}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#b45309")

    # 엘보 마커 (배관 70% 지점)
    if n_el > 0:
        t_el = 0.70
        ex = x_pipe_start + t_el * (x_pipe_end - x_pipe_start)
        ey = y1 + t_el * (y2 - y1)
        ax.scatter([ex], [ey], marker="s", s=85, color="#10b981", edgecolor="black", zorder=5)
        ax.text(ex, ey + 0.35, f"90° Elbow x{n_el}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#047857")

    # 출구 연결
    ax.plot([x_pipe_end, x_outlet], [y2, y2], color="#334155", lw=3)
    ax.annotate("", xy=(x_outlet + 0.5, y2), xytext=(x_outlet, y2),
                arrowprops=dict(arrowstyle="->", lw=2.5, color="#0f172a"))

    # 기준선 및 정보 주석
    ax.axhline(y1, color="#94a3b8", linestyle=":", lw=1)
    ax.text(x_inlet, y1 + 0.45, f"Inlet\nP1={p1:.1f} kPa,g", ha="center", va="bottom", fontsize=9)
    ax.text(x_outlet + 0.55, y2, f"Outlet\nTarget P2={p2:.1f} kPa,g", ha="left", va="center", fontsize=9)

    mid_x = 0.5 * (x_pipe_start + x_pipe_end)
    mid_y = 0.5 * (y1 + y2)
    d_text = f"Optimum ID: {display_d:.2f} mm (Old: {current_d:.2f} mm)" if is_replacement else f"Pipe ID: {display_d:.2f} mm"
    ax.text(
        mid_x, min(y1, y2) - 0.55,
        f"{d_text} | L={l_m:.1f} m | Δz={dz:+.1f} m",
        ha="center", va="top", fontsize=9.5, fontweight="bold", color=pipe_color
    )

    ax.set_title("5조 최적 배관 계통 모식도 (Schematic)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0.2, 11.8)
    ax.set_ylim(min(0.3, min(y1, y2) - 1.2), max(3.6, max(y1, y2) + 1.1))
    ax.axis("off")
    plt.tight_layout()
    return fig


st.subheader("[최적 배관 계통 모식도]")
opt_d_val = float(selected["D_mm"]) if selected is not None else None
add_hp_val = float(selected["Additional_Hp_m"]) if selected is not None else float(current["Additional_Hp_m"])

fig_sch = plot_piping_schematic(
    current_d=current_d_mm,
    opt_d=opt_d_val,
    is_replacement=replacement,
    dz=dz_val,
    l_m=length_val,
    n_el=elbow_val,
    n_val=valve_val,
    add_hp=add_hp_val,
    p1=p1_val,
    p2=p2_val
)
st.pyplot(fig_sch)
plt.close(fig_sch)
