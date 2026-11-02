import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import warnings
from scipy.optimize import minimize_scalar

# 경고 메시지 숨기기 설정
warnings.filterwarnings('ignore', category=UserWarning)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 페이지 설정
st.set_page_config(
    page_title="7팀 복합 배관계 및 펌프 통합 해석 프로그램",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ 7팀 복합 배관계 해석 및 펌프/최적화 통합 프로그램")
st.markdown("Darcy-Weisbach 방정식과 Colebrook-White 공식을 활용한 배관 시스템 해석, 펌프 효율 분석 및 최적화 툴입니다.")
st.markdown("---")

# ==========================================
# 1. 계산 엔진 함수 정의
# ==========================================
def calculate_system(q_m3h, p1_kpa, d_m, l_m, eps_mm, rho, mu, z_diff, n_el, k_el, n_val, k_val, pump_type, n_pump):
    if q_m3h <= 0 or d_m <= 0 or n_pump <= 0:
        return {
            "Q_m3h": 0.0, "Velocity": 0.0, "Reynolds": 0.0, "Friction_Factor": 0.0,
            "Major_Loss": 0.0, "Minor_Loss": 0.0, "Total_Loss": 0.0, "P2_kPa": p1_kpa,
            "Major_Ratio": 0.0, "Minor_Ratio": 0.0, "Pump_Head": 0.0, "Pump_Efficiency": 0.0,
            "Hydraulic_Power_kW": 0.0, "Shaft_Power_kW": 0.0
        }

    q_single = q_m3h / n_pump
    Q = q_m3h / 3600.0
    area = np.pi * (d_m ** 2) / 4.0
    V = Q / area if area > 0 else 0
    reynolds = (rho * V * d_m) / mu if mu > 0 else 0

    eps_rel = (eps_mm * 0.001) / d_m
    if reynolds > 4000:
        f = 0.25 / (np.log10(eps_rel / 3.7 + 5.74 / (reynolds ** 0.9))) ** 2
    elif reynolds > 0:
        f = 64.0 / reynolds
    else:
        f = 0.0

    g = 9.81
    h_major = f * (l_m / d_m) * (V ** 2) / (2 * g)
    sum_k = (n_el * k_el) + (n_val * k_val)
    h_minor = sum_k * (V ** 2) / (2 * g)
    h_total = h_major + h_minor

    pump_specs = {
        "표준 원심펌프 (Standard Centrifugal)": {"max_eff": 0.75, "bep_q": 5.0, "h_base": 20.0},
        "고효율 원심펌프 (High-Efficiency)": {"max_eff": 0.88, "bep_q": 6.0, "h_base": 25.0},
        "다단 터빈 펌프 (Multi-stage Turbine)": {"max_eff": 0.82, "bep_q": 4.5, "h_base": 35.0}
    }
    spec = pump_specs.get(pump_type, pump_specs["표준 원심펌프 (Standard Centrifugal)"])
    
    h_pump = h_total - z_diff if (h_total - z_diff) > 0 else 5.0
    
    q_ratio = q_single / spec["bep_q"]
    eff = spec["max_eff"] * (1.0 - 0.5 * (q_ratio - 1.0) ** 2)
    eff = max(0.2, min(eff, spec["max_eff"]))

    p1_pa = p1_kpa * 1000.0
    p2_pa = p1_pa + (rho * g * z_diff) + (rho * g * (h_pump * n_pump)) - (rho * g * h_total)

    hyd_power = (rho * g * Q * h_total) / 1000.0
    shaft_power = hyd_power / eff if eff > 0 else 0.0

    major_ratio = (h_major / h_total * 100) if h_total > 0 else 0.0
    minor_ratio = (h_minor / h_total * 100) if h_total > 0 else 0.0

    return {
        "Q_m3h": q_m3h, "Velocity": V, "Reynolds": reynolds, "Friction_Factor": f,
        "Major_Loss": h_major, "Minor_Loss": h_minor, "Total_Loss": h_total,
        "P2_kPa": p2_pa / 1000.0, "Major_Ratio": major_ratio, "Minor_Ratio": minor_ratio,
        "Pump_Head": h_pump, "Pump_Efficiency": eff * 100.0,
        "Hydraulic_Power_kW": hyd_power, "Shaft_Power_kW": shaft_power
    }

# ==========================================
# 2. 사이드바 입력 위젯 설정
# ==========================================
st.sidebar.header("🎛️ 운전 및 배관 조건 입력")

q = st.sidebar.number_input("목표유량 (m³/h)", value=5.0, step=0.5)
p1 = st.sidebar.number_input("입구압력 (kPa)", value=150.0, step=10.0)
pump_type = st.sidebar.selectbox(
    "펌프 종류",
    ['표준 원심펌프 (Standard Centrifugal)', '고효율 원심펌프 (High-Efficiency)', '다단 터빈 펌프 (Multi-stage Turbine)']
)
n_pump = st.sidebar.number_input("운전 펌프 대수", value=1, min_value=1, step=1)

length = st.sidebar.number_input("직관총길이 (m)", value=30.0, step=1.0)
od = st.sidebar.number_input("외경 OD (mm)", value=33.4, step=1.0)
t = st.sidebar.number_input("벽두께 t (mm)", value=3.38, step=0.1)
eps = st.sidebar.number_input("배관조도 ε (mm)", value=0.045, step=0.005, format="%.3f")

rho = st.sidebar.number_input("밀도 (kg/m³)", value=1000.0, step=10.0)
mu = st.sidebar.number_input("점성 (Pa·s)", value=0.001, step=0.0001, format="%.4f")
z_diff = st.sidebar.number_input("고도차 z2-z1 (m)", value=2.0, step=0.5)

elbow = st.sidebar.number_input("엘보 개수 (90°)", value=2, step=1)
k_el = st.sidebar.number_input("엘보 K값", value=0.75, step=0.05)
valve = st.sidebar.number_input("밸브 개수", value=1, step=1)
k_val = st.sidebar.number_input("밸브 K값", value=5.0, step=0.5)

# 내경 계산
d_mm = od - (2 * t)
D = d_mm / 1000.0 if d_mm > 0 else 0.0266

# ==========================================
# 3. 메인 화면 출력 (결과 요약 및 최적화)
# ==========================================
res = calculate_system(q, p1, D, length, eps, rho, mu, z_diff, elbow, k_el, valve, k_val, pump_type, n_pump)

st.subheader("📊 해석 및 펌프 효율 분석 결과 요약")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("관 내경 (D)", f"{d_mm:.2f} mm")
    st.metric("평균 유속 (V)", f"{res['Velocity']:.3f} m/s")
with col2:
    st.metric("총 손실수두 (h_L)", f"{res['Total_Loss']:.3f} m")
    st.metric("레이놀즈 수 (Re)", f"{res['Reynolds']:,.0f}")
with col3:
    st.metric("펌프 운전 효율", f"{res['Pump_Efficiency']:.1f} %")
    st.metric("펌프 공급 양정", f"{res['Pump_Head']:.2f} m")
with col4:
    st.metric("소비전력 (축동력)", f"{res['Shaft_Power_kW']:.3f} kW")
    st.metric("최종 출구압력 (P2)", f"{res['P2_kPa']:.1f} kPa")

st.markdown("---")

# 최적화 수행
def objective_func(d_opt_m):
    if d_opt_m <= 0:
        return 1e9
    temp_res = calculate_system(q, p1, d_opt_m, length, eps, rho, mu, z_diff, elbow, k_el, valve, k_val, pump_type, n_pump)
    return temp_res["Total_Loss"]

min_bound = max(0.005, D * 0.5)
max_bound = max(0.2, D * 3.0)
opt_result = minimize_scalar(objective_func, bounds=(min_bound, max_bound), method='bounded')

st.subheader("💡 배관 직경 최적화 결과 (최소 손실수두)")
if opt_result.success:
    opt_D = opt_result.x
    opt_d_mm = opt_D * 1000.0
    opt_res = calculate_system(q, p1, opt_D, length, eps, rho, mu, z_diff, elbow, k_el, valve, k_val, pump_type, n_pump)
    
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        st.metric("최적 추천 관 내경", f"{opt_d_mm:.2f} mm")
    with col_opt2:
        st.metric("최적화 후 총 손실수두", f"{opt_res['Total_Loss']:.4f} m", f"{opt_res['Total_Loss'] - res['Total_Loss']:.3f} m")
    with col_opt3:
        st.metric("최적화 후 축동력", f"{opt_res['Shaft_Power_kW']:.3f} kW", f"{opt_res['Shaft_Power_kW'] - res['Shaft_Power_kW']:.3f} kW")
else:
    st.warning("최적화 수렴에 실패했습니다.")

st.markdown("---")

# ==========================================
# 4. 시각화 (그래프 3개)
# ==========================================
st.subheader("📈 유량 변화에 따른 시스템 성능 변화 곡선")

sweep_flows = np.linspace(1.0, 20.0, 25)
sweep_res = [calculate_system(fq, p1, D, length, eps, rho, mu, z_diff, elbow, k_el, valve, k_val, pump_type, n_pump) for fq in sweep_flows]

fq_vals = [r["Q_m3h"] for r in sweep_res]
h_tot = [r["Total_Loss"] for r in sweep_res]
eff_vals = [r["Pump_Efficiency"] for r in sweep_res]
power_vals = [r["Shaft_Power_kW"] for r in sweep_res]

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

axes[0].plot(fq_vals, h_tot, 'k-', lw=2)
axes[0].set_title("1. Flow vs Total Head Loss")
axes[0].set_xlabel("Flow Rate (m³/h)")
axes[0].set_ylabel("Head Loss (m)")
axes[0].grid(True)

axes[1].plot(fq_vals, eff_vals, 'b-o', lw=1.5)
axes[1].set_title("2. Flow vs Pump Efficiency (%)")
axes[1].set_xlabel("Flow Rate (m³/h)")
axes[1].set_ylabel("Efficiency (%)")
axes[1].grid(True)

axes[2].plot(fq_vals, power_vals, 'r-s', lw=1.5)
axes[2].set_title("3. Flow vs Shaft Power (kW)")
axes[2].set_xlabel("Flow Rate (m³/h)")
axes[2].set_ylabel("Power (kW)")
axes[2].grid(True)

plt.tight_layout()
st.pyplot(fig)
