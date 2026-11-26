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
