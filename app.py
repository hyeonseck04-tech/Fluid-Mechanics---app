import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

plt.rcParams["axes.unicode_minus"] = False

G = 9.81
STYLE = {"description_width": "initial"}


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
# 3. 입력 위젯
# ==========================================
def float_input(value, description):
    return widgets.FloatText(
        value=value,
        description=description,
        style=STYLE
    )


def int_input(value, description):
    return widgets.IntText(
        value=value,
        description=description,
        style=STYLE
    )


q_input = float_input(5.0, "목표유량(m³/h):")
p1_input = float_input(150.0, "공급압력(kPa,g):")
p2_input = float_input(0.0, "목표 출구압력(kPa,g):")
dz_input = float_input(2.0, "고도차(z2-z1,m):")

rho_input = float_input(1000.0, "밀도(kg/m³):")
mu_input = float_input(0.001, "점도(Pa·s):")

length_input = float_input(30.0, "직관 총길이(m):")
od_input = float_input(33.4, "기존 외경 OD(mm):")
t_input = float_input(3.38, "기존 벽두께 t(mm):")
eps_input = float_input(0.045, "배관조도(mm):")

elbow_input = int_input(2, "90° 엘보 개수:")
kel_input = float_input(0.75, "엘보 K값:")
valve_input = int_input(1, "밸브 개수:")
kval_input = float_input(5.0, "밸브 K값:")

mode_input = widgets.Dropdown(
    options=["기존 배관 유지", "배관 교체 허용"],
    value="기존 배관 유지",
    description="설계 조건:",
    style=STYLE
)

dmin_input = float_input(20.0, "탐색 최소 내경(mm):")
dmax_input = float_input(80.0, "탐색 최대 내경(mm):")
vmax_input = float_input(3.0, "허용 최대유속(m/s):")
hpmax_input = float_input(20.0, "추가 펌프수두 한도(m):")

all_inputs = [
    q_input, p1_input, p2_input, dz_input,
    rho_input, mu_input, length_input,
    od_input, t_input, eps_input,
    elbow_input, kel_input, valve_input, kval_input,
    mode_input, dmin_input, dmax_input,
    vmax_input, hpmax_input
]

ui = widgets.VBox([
    widgets.HTML("<h3>7팀 복합 배관계 해석 프로그램</h3>"),
    widgets.HTML(
        "압력 기준: 게이지압 | "
        "입구·출구: 동일 내경의 관 단면 | "
        "마찰계수: Darcy 기준"
    ),
    widgets.HBox([q_input, p1_input]),
    widgets.HBox([p2_input, dz_input]),
    widgets.HBox([rho_input, mu_input]),
    widgets.HBox([length_input, eps_input]),
    widgets.HBox([od_input, t_input]),
    widgets.HBox([elbow_input, kel_input]),
    widgets.HBox([valve_input, kval_input]),
    widgets.HTML("<h4>관경 탐색 및 제한조건</h4>"),
    mode_input,
    widgets.HBox([dmin_input, dmax_input]),
    widgets.HBox([vmax_input, hpmax_input]),
    widgets.HTML(
        "선정 기준: 제한조건을 만족하는 최소 탐색 내경<br>"
        "탐색 방식: 연속 내경 후보 301개, 표준 규격 미적용<br>"
        "교체 조건: 길이·조도·부속 개수·K값 동일<br>"
        "전이영역: 2000 ≤ Re ≤ 4000, 손실계산 및 선정 제외"
    )
])

out = widgets.Output()


# ==========================================
# 4. 계산 및 결과 출력
# ==========================================
def update_results(change=None):
    with out:
        clear_output(wait=True)

        replacement = mode_input.value == "배관 교체 허용"
        dmin_input.disabled = not replacement
        dmax_input.disabled = not replacement

        numeric_widgets = [
            w for w in all_inputs if w is not mode_input
        ]

        if not all(np.isfinite(w.value) for w in numeric_widgets):
            print("입력 오류: 유한하지 않은 숫자")
            return

        current_d_mm = od_input.value - 2.0 * t_input.value

        if (
            q_input.value <= 0
            or rho_input.value <= 0
            or mu_input.value <= 0
            or od_input.value <= 0
            or t_input.value < 0
            or current_d_mm <= 0
        ):
            print(
                "입력 오류: 유량·밀도·점도·외경·내경은 양수, "
                "벽두께는 0 이상"
            )
            return

        nonnegative = [
            length_input, eps_input, elbow_input,
            kel_input, valve_input, kval_input
        ]

        if any(w.value < 0 for w in nonnegative):
            print("입력 오류: 길이·조도·부속 개수·K값에 음수 포함")
            return

        if vmax_input.value <= 0 or hpmax_input.value < 0:
            print(
                "입력 오류: 최대유속은 양수, "
                "추가 펌프수두 한도는 0 이상"
            )
            return

        if replacement and not (
            0 < dmin_input.value < dmax_input.value
        ):
            print("입력 오류: 탐색 범위는 0 < 최소 내경 < 최대 내경")
            return

        def run(
            d_mm, q=None, length=None,
            kel=None, kval=None, method="sj"
        ):
            return calculate_system(
                q_m3h=q_input.value if q is None else q,
                p1_kpa=p1_input.value,
                d_m=d_mm / 1000.0,
                l_m=length_input.value if length is None else length,
                eps_mm=eps_input.value,
                rho=rho_input.value,
                mu=mu_input.value,
                z_diff=dz_input.value,
                n_el=elbow_input.value,
                k_el=kel_input.value if kel is None else kel,
                n_val=valve_input.value,
                k_val=kval_input.value if kval is None else kval,
                p2_target_kpa=p2_input.value,
                method=method
            )

        current = run(current_d_mm)

        print("=" * 65)
        print("[기존 배관 해석]")
        print(f"관 내경: {current_d_mm:.3f} mm")
        print(f"평균 유속: {current['Velocity']:.3f} m/s")
        print(f"Reynolds 수: {current['Reynolds']:,.0f}")
        print(f"유동 상태: {current['Regime']}")

        if current["Transition"]:
            print("손실수두·압력 계산: 전이영역으로 제외")
        else:
            print(f"Darcy 마찰계수: {current['Friction_Factor']:.6f}")
            print(f"직관 손실수두: {current['Major_Loss']:.3f} m")
            print(f"엘보 손실수두: {current['Elbow_Loss']:.3f} m")
            print(f"밸브 손실수두: {current['Valve_Loss']:.3f} m")
            print(f"총 손실수두: {current['Total_Loss']:.3f} m")
            print(
                f"주손실 / 부손실: "
                f"{current['Major_Ratio']:.1f}% / "
                f"{current['Minor_Ratio']:.1f}%"
            )
            print(
                f"추가 펌프 없는 출구압력: "
                f"{current['P2_kPa']:.3f} kPa,g"
            )
            print(
                f"목표 출구압력 기준 필요 입구압력: "
                f"{current['Required_P1_kPa']:.3f} kPa,g"
            )
            print(
                f"추가 펌프 필요수두: "
                f"{current['Additional_Hp_m']:.3f} m"
            )

        # ------------------------------------------
        # 관경 탐색
        # ------------------------------------------
        diameters = (
            np.linspace(dmin_input.value, dmax_input.value, 301)
            if replacement else np.array([current_d_mm])
        )

        candidates = pd.DataFrame([
            run(d) for d in diameters
        ])

        candidates["Feasible"] = (
            (~candidates["Transition"])
            & np.isfinite(candidates["Total_Loss"])
            & (candidates["Velocity"] <= vmax_input.value)
            & (
                candidates["Additional_Hp_m"]
                <= hpmax_input.value
            )
        )

        feasible = candidates[candidates["Feasible"]]
        selected = None

        print("\n[관경 선정]")
        print(f"설계 조건: {mode_input.value}")
        print(f"허용 최대유속: {vmax_input.value:.3f} m/s")
        print(
            f"추가 펌프수두 한도: "
            f"{hpmax_input.value:.3f} m"
        )

        if replacement:
            spacing = (
                dmax_input.value - dmin_input.value
            ) / 300.0

            print(
                f"탐색 내경: {dmin_input.value:.3f}"
                f" ~ {dmax_input.value:.3f} mm"
            )
            print(f"후보 개수: {len(candidates)}개")
            print(f"탐색 간격: {spacing:.3f} mm")

        if feasible.empty:
            print("선정 결과: 제한조건을 만족하는 후보 없음")
        else:
            selected = feasible.iloc[0]

            if replacement:
                print(
                    f"최소 적합 탐색 내경: "
                    f"{selected['D_mm']:.3f} mm"
                )
            else:
                print("선정 결과: 기존 배관 제한조건 충족")

            print(
                f"선정 유속: "
                f"{selected['Velocity']:.3f} m/s"
            )
            print(
                f"선정 총 손실수두: "
                f"{selected['Total_Loss']:.3f} m"
            )
            print(
                f"선정 추가 펌프 필요수두: "
                f"{selected['Additional_Hp_m']:.3f} m"
            )

            if selected["Additional_Hp_m"] == 0:
                print(
                    "압력 조건: 기존 공급압력으로 "
                    "목표 출구압력 이상 확보 가능"
                )
                print(
                    "유량 조건: 목표유량 고정, "
                    "실제 유량조절 동작은 계산 범위 외"
                )

        columns = [
            "D_mm", "Velocity", "Reynolds",
            "Total_Loss", "Additional_Hp_m", "Feasible"
        ]

        table = (
            feasible.head(5)
            if not feasible.empty else candidates.head(5)
        )

        table = table[columns].rename(columns={
            "D_mm": "내경(mm)",
            "Velocity": "유속(m/s)",
            "Reynolds": "Re",
            "Total_Loss": "총 손실수두(m)",
            "Additional_Hp_m": "추가 펌프수두(m)",
            "Feasible": "조건 충족"
        })

        print("\n[후보 계산표: 최대 5개]")
        display(table.round(3))

        # ------------------------------------------
        # 상대차 및 극한조건 검증
        # ------------------------------------------
        check_d = (
            float(selected["D_mm"])
            if selected is not None else current_d_mm
        )

        check = run(check_d)

        print(f"\n[계산 검증: 내경 {check_d:.3f} mm]")

        if check["Transition"]:
            print("검증 결과: 전이영역으로 계산 제외")
        else:
            reference = run(check_d, method="colebrook")

            actual = check["Total_Loss"]
            expected = reference["Total_Loss"]

            print(f"프로그램 총 손실수두: {actual:.6f} m")
            print(f"기준계산 총 손실수두: {expected:.6f} m")

            if expected > 0:
                error = (
                    abs(actual - expected)
                    / abs(expected) * 100.0
                )
                print(f"기준계산 대비 상대차: {error:.4f}%")
            else:
                print(
                    f"기준계산 대비 절대차: "
                    f"{abs(actual - expected):.6f} m"
                )

            if check["Reynolds"] > 4000:
                print(
                    "비교 기준: Swamee–Jain 근사식 / "
                    "Colebrook 반복계산"
                )
                print("검증 구분: 계산법 간 비교, 실험값 비교 아님")
            else:
                print("비교 기준: 층류 Darcy 마찰계수 64/Re")

            k_zero = run(check_d, kel=0.0, kval=0.0)
            l_zero = run(check_d, length=0.0)

            print(
                f"K=0 총 손실수두: "
                f"{k_zero['Total_Loss']:.6f} m"
            )
            print(
                f"L=0 총 손실수두: "
                f"{l_zero['Total_Loss']:.6f} m"
            )
            print(
                "K=0 검증 절대차: "
                f"{abs(k_zero['Total_Loss'] - check['Major_Loss']):.3e} m"
            )
            print(
                "L=0 검증 절대차: "
                f"{abs(l_zero['Total_Loss'] - check['Minor_Loss']):.3e} m"
            )

        print("=" * 65)

        # ------------------------------------------
        # 유량 조건 자동 반복계산
        # ------------------------------------------
        flow_max = max(20.0, 1.2 * q_input.value)
        flows = np.linspace(0.0, flow_max, 41)

        flow_df = pd.DataFrame([
            run(current_d_mm, q=q) for q in flows
        ])

        fig, axes = plt.subplots(2, 3, figsize=(17, 9))

        # 1. 유량-손실수두
        ax = axes[0, 0]
        ax.plot(
            flows, flow_df["Total_Loss"],
            "k-", label="Total"
        )
        ax.plot(
            flows, flow_df["Major_Loss"],
            "b--", label="Major"
        )
        ax.plot(
            flows, flow_df["Minor_Loss"],
            "r:", label="Minor"
        )
        ax.set_title("Existing Pipe: Flow vs Head Loss")
        ax.set_xlabel("Flow rate (m³/h)")
        ax.set_ylabel("Head loss (m)")
        ax.legend()
        ax.grid(True)

        # 2. 유량-Reynolds 수
        ax = axes[0, 1]
        ax.plot(
            flows, flow_df["Reynolds"],
            color="green"
        )
        ax.axhspan(
            2000, 4000,
            color="orange", alpha=0.25,
            label="Transition"
        )
        ax.set_title("Existing Pipe: Flow vs Reynolds")
        ax.set_xlabel("Flow rate (m³/h)")
        ax.set_ylabel("Reynolds number")
        ax.legend()
        ax.grid(True)

        # 3. 손실 비율
        ax = axes[0, 2]

        if (
            np.isfinite(current["Total_Loss"])
            and current["Total_Loss"] > 0
        ):
            ax.pie(
                [
                    current["Major_Loss"],
                    current["Minor_Loss"]
                ],
                labels=["Major", "Minor"],
                autopct="%1.1f%%",
                startangle=90
            )
        else:
            ax.text(
                0.5, 0.5,
                "No valid positive loss",
                ha="center", va="center",
                transform=ax.transAxes
            )

        ax.set_title("Existing Pipe: Loss Ratio")

        # 4. 관경-유속
        ax = axes[1, 0]
        ax.plot(
            candidates["D_mm"],
            candidates["Velocity"],
            "b.-"
        )
        ax.axhline(
            vmax_input.value,
            color="r", linestyle="--",
            label="Velocity limit"
        )
        ax.set_title("Diameter vs Velocity")
        ax.set_xlabel("Internal diameter (mm)")
        ax.set_ylabel("Velocity (m/s)")
        ax.legend()
        ax.grid(True)

        # 5. 관경-추가 펌프수두
        ax = axes[1, 1]
        ax.plot(
            candidates["D_mm"],
            candidates["Additional_Hp_m"],
            "k.-"
        )
        ax.axhline(
            hpmax_input.value,
            color="r", linestyle="--",
            label="Pump head limit"
        )
        ax.set_title("Diameter vs Additional Pump Head")
        ax.set_xlabel("Internal diameter (mm)")
        ax.set_ylabel("Additional pump head (m)")
        ax.legend()
        ax.grid(True)

        if selected is not None:
            for ax in [axes[1, 0], axes[1, 1]]:
                ax.axvline(
                    selected["D_mm"],
                    color="green", linestyle=":"
                )

        # 6. 밸브 K-손실수두
        k_values = np.linspace(
            0.0,
            max(10.0, 2.0 * kval_input.value),
            25
        )

        k_results = [
            run(current_d_mm, kval=k)
            for k in k_values
        ]

        ax = axes[1, 2]
        ax.plot(
            k_values,
            [r["Total_Loss"] for r in k_results],
            label="Total loss"
        )
        ax.plot(
            k_values,
            [r["Minor_Loss"] for r in k_results],
            "--", label="Minor loss"
        )
        ax.set_title("Existing Pipe: Valve K vs Loss")
        ax.set_xlabel("K per valve (valve count fixed)")
        ax.set_ylabel("Head loss (m)")
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()
        plt.close(fig)


# ==========================================
# 5. 이벤트 연결 및 초기 실행
# ==========================================
for w in all_inputs:
    w.observe(update_results, names="value")

display(ui, out)
update_results()
