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
# 6. 실제 배관 계통도 (P&ID 스타일 입상/직각 배관 구조)
# ==========================================
def draw_valve_symbol(ax, x, y, horizontal=True, label=None):
    """P&ID 표준 게이트/글로브 밸브 심볼 (모래시계 + 핸들)"""
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
    """P&ID 표준 원심펌프 심볼 (원형 케이싱 + 받침대 + 접선 토출 노즐)"""
    r = 0.42
    color = "#dc2626" if active else "#64748b"
    fill = "#fef2f2" if active else "#f8fafc"

    # 받침대 (Baseplate)
    base = plt.Polygon([[cx - r * 0.85, cy - r * 1.35], [cx + r * 0.85, cy - r * 1.35],
                        [cx + r * 0.55, cy - r * 0.75], [cx - r * 0.55, cy - r * 0.75]],
                       closed=True, facecolor=color, edgecolor="#1e293b", lw=1.2, zorder=4)
    ax.add_patch(base)

    # 펌프 케이싱 원
    circle = plt.Circle((cx, cy), r, facecolor=fill, edgecolor=color, lw=2.5, zorder=5)
    ax.add_patch(circle)

    # 접선 토출 노즐 (오른쪽 상단)
    ax.plot([cx, cx + r * 1.35], [cy + r, cy + r], color=color, lw=3.2, zorder=4)

    # 임펠러 회전 방향 화살표/표기
    ax.text(cx, cy, "P-101", ha="center", va="center", fontsize=8.5, fontweight="bold", color=color, zorder=6)
    status = f"PUMP ON (+{add_hp:.2f} m)" if active else "BYPASS / NO EXTRA PUMP"
    ax.text(cx, cy - r * 1.55, status, ha="center", va="top", fontsize=8.5, fontweight="bold", color=color)


def draw_tank(ax, x, y, width=1.1, height=1.5, title="TANK", info=""):
    """공정 탱크(저장조) 및 수면 심볼"""
    rect = plt.Rectangle((x - width / 2, y - height / 2), width, height,
                         facecolor="#f1f5f9", edgecolor="#334155", lw=2, zorder=3)
    ax.add_patch(rect)
    # 내부 유체 영역
    fluid_h = height * 0.65
    fluid = plt.Rectangle((x - width / 2 + 0.03, y - height / 2 + 0.03), width - 0.06, fluid_h,
                          facecolor="#bae6fd", edgecolor="none", zorder=2)
    ax.add_patch(fluid)
    # 수면 파형 선
    wx = np.linspace(x - width / 2 + 0.05, x + width / 2 - 0.05, 30)
    wy = (y - height / 2 + fluid_h) + 0.04 * np.sin(np.linspace(0, 3 * np.pi, 30))
    ax.plot(wx, wy, color="#0284c7", lw=1.5, zorder=4)

    ax.text(x, y + height / 2 + 0.15, title, ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#0f172a")
    if info:
        ax.text(x, y - height / 2 - 0.18, info, ha="center", va="top", fontsize=8.5, color="#334155")


def plot_real_piping_diagram(current_d, opt_d, is_replacement, dz, l_m, n_el, n_val, add_hp, p1, p2, q_m3h):
    fig, ax = plt.subplots(figsize=(14, 5.6))

    display_d = opt_d if (is_replacement and opt_d is not None) else current_d
    pipe_lw = np.clip(display_d / 10.0, 2.8, 8.5)
    pipe_color = "#0284c7" if is_replacement else "#1e293b"
    pump_active = bool(np.isfinite(add_hp) and add_hp > 1e-6)

    y_base = 1.8
    # 고도차 반영 (최소 시각 단차 확보)
    visual_dz = np.clip(dz * 0.35, -1.0, 2.2)
    if abs(dz) > 0 and abs(visual_dz) < 0.7:
        visual_dz = 0.7 if dz > 0 else -0.7
    y_top = y_base + visual_dz

    # 1. 입구 탱크 & 흡입 배관 & 펌프
    draw_tank(ax, 1.0, y_base + 0.25, title="SUCTION TANK", info=f"P1 = {p1:.1f} kPa,g\nz1 = 0.0 m")
    ax.plot([1.55, 2.7], [y_base, y_base], color="#334155", lw=pipe_lw, zorder=3)
    draw_centrifugal_pump(ax, 3.1, y_base, active=pump_active, add_hp=add_hp)

    # 펌프 토출 노즐 높이에서 시작
    y_disch = y_base + 0.42
    x_start = 3.65
    x_end = 10.6

    # 2. 엘보 개수(n_el)에 따른 실제 90도 직각 배관 경로 생성
    pts = [(x_start, y_disch)]
    if n_el <= 0:
        # 엘보 0개: 수평 직관 연결 (고도차 없을 때 완전 직선)
        pts.append((x_end, y_top))
    elif n_el == 1:
        # 90도 엘보 1개: 수평 이동 후 수직 상승/하강
        pts.append((x_end, y_disch))
        pts.append((x_end, y_top))
    elif n_el == 2:
        # 90도 엘보 2개: 수평 -> 수직 입상(Riser) -> 상부 수평관
        x_mid = 6.6
        pts.append((x_mid, y_disch))
        pts.append((x_mid, y_top))
        pts.append((x_end, y_top))
    else:
        # 90도 엘보 3개 이상: 입력된 엘보 개수만큼 계단식 직각 꺾임 배치
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

    # 배관 외벽(진한 색) + 내부 유체 흐름선(밝은 색) 이중선으로 실제 파이프 표현
    px = [p[0] for p in pts]
    py = [p[1] for p in pts]
    ax.plot(px, py, color="#0f172a", lw=pipe_lw + 2.2, solid_joinstyle="miter", zorder=3)
    ax.plot(px, py, color=pipe_color, lw=pipe_lw, solid_joinstyle="miter", zorder=4)

    # 90도 엘보 위치마다 플랜지/피팅 마커 및 번호 표시
    elbow_coords = pts[1:-1]
    for idx, (ex, ey) in enumerate(elbow_coords, start=1):
        ax.scatter([ex], [ey], s=55, color="#10b981", edgecolor="#0f172a", zorder=6)
        offset_y = 0.22 if idx % 2 == 1 else -0.28
        ax.text(ex, ey + offset_y, f"EL-{idx}", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#047857")

    # 3. 밸브 개수(n_val)만큼 실제 배관 구간 위에 밸브 심볼 배치
    if n_val > 0:
        # 가장 긴 수평 구간들을 찾아 밸브 균등 배치
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

    # 4. 출구 탱크 연결 및 유량 화살표
    last_x, last_y = pts[-1]
    ax.plot([last_x, 11.45], [last_y, last_y], color="#0f172a", lw=pipe_lw + 2.2, zorder=3)
    ax.plot([last_x, 11.45], [last_y, last_y], color=pipe_color, lw=pipe_lw, zorder=4)
    draw_tank(ax, 12.0, last_y + 0.25, title="DISCHARGE TANK", info=f"Target P2 = {p2:.1f} kPa,g\nz2 = {dz:+.1f} m")

    # 유량 방향 화살표 (흡입관 & 토출관)
    ax.annotate("", xy=(2.45, y_base + 0.25), xytext=(1.75, y_base + 0.25),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#0284c7"))
    ax.text(2.1, y_base + 0.35, f"Q={q_m3h:.2f} m³/h", ha="center", va="bottom", fontsize=8, color="#0284c7", fontweight="bold")

    # 치수선 (고도차 Δz 및 총 직관길이 L)
    ax.plot([10.9, 10.9], [y_base, last_y], color="#64748b", linestyle="--", lw=1.2)
    ax.annotate("", xy=(10.9, last_y), xytext=(10.9, y_base),
                arrowprops=dict(arrowstyle="<->", lw=1.2, color="#475569"))
    ax.text(11.0, (y_base + last_y) / 2, f"Δz = {dz:+.2f} m", ha="left", va="center", fontsize=8.5, fontweight="bold", color="#334155")

    # 하단 배관 사양 박스
    spec_title = f"OPTIMUM PIPE ID: {display_d:.2f} mm (Existing: {current_d:.2f} mm)" if is_replacement else f"PIPE ID: {display_d:.2f} mm (Existing Kept)"
    spec_detail = f"Total Straight Length L = {l_m:.1f} m  |  90° Elbows: {n_el} ea  |  Valves: {n_val} ea"
    ax.text(6.5, 0.45, f"{spec_title}\n{spec_detail}",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8fafc", edgecolor=pipe_color, lw=1.8))

    ax.set_xlim(0.1, 13.0)
    ax.set_ylim(-0.2, max(4.8, last_y + 1.5))
    ax.axis("off")
    plt.tight_layout()
    return fig


st.subheader("[최적 배관 계통도 (실제 배관 배치 구조)]")
opt_d_val = float(selected["D_mm"]) if selected is not None else None
add_hp_val = float(selected["Additional_Hp_m"]) if selected is not None else float(current["Additional_Hp_m"])

fig_sch = plot_real_piping_diagram(
    current_d=current_d_mm,
    opt_d=opt_d_val,
    is_replacement=replacement,
    dz=dz_val,
    l_m=length_val,
    n_el=elbow_val,
    n_val=valve_val,
    add_hp=add_hp_val,
    p1=p1_val,
    p2=p2_val,
    q_m3h=q_val
)
st.pyplot(fig_sch)
plt.close(fig_sch)
