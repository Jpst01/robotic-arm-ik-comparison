import argparse
import numpy as np

SAVE_ONLY = False


def _init_matplotlib():
    global plt, animation
    if SAVE_ONLY:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import matplotlib.animation as _anim
    plt = _plt
    animation = _anim


L1 = 0.3
L2 = 0.25
L3 = 0.15


def forward_kinematics(t1, t2, t3):
    base = np.array([0.0, 0.0, 0.0])
    shoulder = np.array([0.0, 0.0, L1])

    r_elbow = L2 * np.cos(t2)
    elbow = np.array([
        np.cos(t1) * r_elbow,
        np.sin(t1) * r_elbow,
        L1 + L2 * np.sin(t2),
    ])

    r_ee = L2 * np.cos(t2) + L3 * np.cos(t2 + t3)
    ee = np.array([
        np.cos(t1) * r_ee,
        np.sin(t1) * r_ee,
        L1 + L2 * np.sin(t2) + L3 * np.sin(t2 + t3),
    ])

    return base, shoulder, elbow, ee


def analytical_ik(x, y, z):
    t1 = np.arctan2(y, x)
    r = np.sqrt(x**2 + y**2)
    zp = z - L1
    d2 = r**2 + zp**2
    d = np.sqrt(d2)

    if d > L2 + L3 or d < abs(L2 - L3):
        return None

    cos_t3 = (d2 - L2**2 - L3**2) / (2.0 * L2 * L3)
    cos_t3 = np.clip(cos_t3, -1.0, 1.0)
    t3 = np.arccos(cos_t3)

    alpha = np.arctan2(zp, r)
    beta = np.arctan2(L3 * np.sin(t3), L2 + L3 * np.cos(t3))
    t2 = alpha - beta

    return t1, t2, t3


def smoothstep_trajectory(start, goal, steps=60):
    trajectory = []
    for i in range(steps + 1):
        t = i / steps
        s = 3 * t**2 - 2 * t**3
        q = start + s * (goal - start)
        trajectory.append(q)
    return trajectory


def draw_arm(ax, t1, t2, t3, color="royalblue", alpha=1.0, linewidth=3):
    base, shoulder, elbow, ee = forward_kinematics(t1, t2, t3)
    pts = np.array([base, shoulder, elbow, ee])

    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            "-o", color=color, linewidth=linewidth,
            markersize=6, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=2, alpha=alpha)

    ax.scatter(*base, s=120, c="gray", marker="s", zorder=5, alpha=alpha)
    ax.scatter(*ee, s=60, c="red", marker="D", zorder=5, alpha=alpha)

    return ee


def setup_3d_axes(ax, title=""):
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([-0.1, 0.8])
    ax.set_xlabel("X (m)", fontsize=10)
    ax.set_ylabel("Y (m)", fontsize=10)
    ax.set_zlabel("Z (m)", fontsize=10)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.view_init(elev=25, azim=135)


def demo_static():
    configs = [
        ("Home",        0.0,        0.0,       0.0),
        ("Reach up",    0.0,        np.pi/2,   0.0),
        ("Reach fwd",   0.0,        0.3,       0.5),
        ("Rotated",     np.pi/3,    0.4,       0.6),
    ]

    fig = plt.figure(figsize=(16, 4))
    fig.suptitle("3-DOF Robotic Arm — Static Configurations",
                 fontsize=14, fontweight="bold")

    for i, (name, t1, t2, t3) in enumerate(configs):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        draw_arm(ax, t1, t2, t3)
        _, _, _, ee = forward_kinematics(t1, t2, t3)
        setup_3d_axes(ax, f"{name}\n({t1:.1f}, {t2:.1f}, {t3:.1f})")
        ax.text(ee[0], ee[1], ee[2] + 0.05,
                f"({ee[0]:.2f}, {ee[1]:.2f}, {ee[2]:.2f})",
                fontsize=7, ha="center")

    plt.tight_layout()
    plt.savefig("assets/arm_static_poses.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/arm_static_poses.png")
    if not SAVE_ONLY:
        plt.show()
    plt.close()


def demo_animated():
    targets = [
        np.array([0.25,  0.0,  0.45]),
        np.array([0.15,  0.15, 0.40]),
        np.array([0.0,   0.30, 0.35]),
        np.array([-0.1,  0.2,  0.50]),
        np.array([0.20, -0.10, 0.38]),
        np.array([0.30,  0.0,  0.30]),
    ]

    all_frames = []
    current_q = np.array([0.0, 0.0, 0.0])

    for target in targets:
        ik = analytical_ik(*target)
        if ik is None:
            continue
        goal_q = np.array(ik)
        frames = smoothstep_trajectory(current_q, goal_q, steps=40)
        all_frames.extend(frames)
        all_frames.extend([goal_q] * 10)
        current_q = goal_q

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    trace_x, trace_y, trace_z = [], [], []

    def update(frame_idx):
        ax.cla()
        setup_3d_axes(ax, "3-DOF Arm — Trajectory Animation")

        q = all_frames[frame_idx]
        _, _, _, ee = forward_kinematics(*q)

        trace_x.append(ee[0])
        trace_y.append(ee[1])
        trace_z.append(ee[2])

        ax.plot(trace_x, trace_y, trace_z, "-", color="orange",
                linewidth=1, alpha=0.6)

        for i, t in enumerate(targets):
            ax.scatter(*t, s=80, c="limegreen", marker="*", zorder=10)
            ax.text(t[0], t[1], t[2] + 0.03, f"T{i+1}", fontsize=8,
                    ha="center", color="green")

        draw_arm(ax, *q)

        ax.text2D(0.02, 0.95,
                  f"t1={q[0]:.2f}  t2={q[1]:.2f}  t3={q[2]:.2f}",
                  transform=ax.transAxes, fontsize=9,
                  verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

    anim = animation.FuncAnimation(
        fig, update, frames=len(all_frames), interval=30, repeat=False)

    anim.save("assets/arm_trajectory.gif", writer="pillow", fps=30)
    print("Saved: assets/arm_trajectory.gif")
    if not SAVE_ONLY:
        plt.show()
    plt.close()


def demo_workspace():
    rng = np.random.default_rng(42)
    n = 5000

    t1 = rng.uniform(-np.pi, np.pi, n)
    t2 = rng.uniform(-np.pi / 2, np.pi / 2, n)
    t3 = rng.uniform(0, np.pi, n)

    points = np.array([forward_kinematics(a, b, c)[3]
                       for a, b, c in zip(t1, t2, t3)])

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                    c=points[:, 2], cmap="viridis", s=2, alpha=0.5)
    fig.colorbar(sc, ax=ax, shrink=0.6, label="Height Z (m)")

    draw_arm(ax, 0, 0, 0, color="red", linewidth=4)

    setup_3d_axes(ax, "Reachable Workspace (~0.7m hemisphere)")
    plt.savefig("assets/arm_workspace.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/arm_workspace.png")
    if not SAVE_ONLY:
        plt.show()
    plt.close()


def check_collision(trajectory, obs_center, obs_radius):
    for idx, q in enumerate(trajectory):
        base, shoulder, elbow, ee = forward_kinematics(*q)
        segments = [(base, shoulder), (shoulder, elbow), (elbow, ee)]
        for a, b in segments:
            ab = b - a
            ap = obs_center - a
            t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12), 0, 1)
            closest = a + t * ab
            dist = np.linalg.norm(closest - obs_center)
            if dist < obs_radius:
                return idx
    return -1


def replan_around_obstacle(start_q, target, obs_center, obs_radius,
                           num_offsets=16, offset_mag=0.12):
    for i in range(num_offsets):
        angle = 2 * np.pi * i / num_offsets
        for dz in [-offset_mag * 0.5, 0.0, offset_mag * 0.5]:
            via_pos = obs_center + np.array([
                offset_mag * np.cos(angle),
                offset_mag * np.sin(angle),
                dz
            ])
            via_ik = analytical_ik(*via_pos)
            if via_ik is None:
                continue
            via_q = np.array(via_ik)

            leg1 = smoothstep_trajectory(start_q, via_q, steps=40)
            if check_collision(leg1, obs_center, obs_radius) != -1:
                continue

            target_ik = analytical_ik(*target)
            if target_ik is None:
                continue
            goal_q = np.array(target_ik)
            leg2 = smoothstep_trajectory(via_q, goal_q, steps=40)
            if check_collision(leg2, obs_center, obs_radius) != -1:
                continue

            return leg1 + leg2, via_pos
    return None, None


def demo_obstacle():
    obs_center = np.array([0.20, 0.10, 0.40])
    obs_radius = 0.06

    start_q = np.array([0.0, 0.0, 0.0])
    target = np.array([0.15, 0.15, 0.45])

    ik_direct = analytical_ik(*target)
    if ik_direct is None:
        print("Target unreachable, skipping obstacle demo")
        return
    goal_q = np.array(ik_direct)
    direct_traj = smoothstep_trajectory(start_q, goal_q, steps=60)

    collision_idx = check_collision(direct_traj, obs_center, obs_radius)

    avoid_traj, waypoint = replan_around_obstacle(
        start_q, target, obs_center, obs_radius)

    fig = plt.figure(figsize=(14, 6))

    u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:20j]
    sx = obs_center[0] + obs_radius * np.cos(u) * np.sin(v)
    sy = obs_center[1] + obs_radius * np.sin(u) * np.sin(v)
    sz = obs_center[2] + obs_radius * np.cos(v)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    setup_3d_axes(ax1, "Direct Path (Collision!)")
    ax1.plot_surface(sx, sy, sz, color="red", alpha=0.3)
    ax1.text(obs_center[0], obs_center[1], obs_center[2] + obs_radius + 0.03,
             "Obstacle", fontsize=9, ha="center", color="red", fontweight="bold")

    direct_ee = np.array([forward_kinematics(*q)[3] for q in direct_traj])
    ax1.plot(direct_ee[:, 0], direct_ee[:, 1], direct_ee[:, 2],
             "-", color="crimson", linewidth=2, alpha=0.8, label="Direct path")

    if collision_idx >= 0:
        col_ee = direct_ee[collision_idx]
        ax1.scatter(*col_ee, s=200, c="red", marker="X", zorder=10)
        ax1.text(col_ee[0] + 0.03, col_ee[1], col_ee[2],
                 "COLLISION", fontsize=9, color="red", fontweight="bold")
        draw_arm(ax1, *direct_traj[collision_idx], color="crimson", alpha=0.7)
    else:
        draw_arm(ax1, *direct_traj[-1], color="crimson", alpha=0.7)

    ax1.scatter(*target, s=120, c="limegreen", marker="*", zorder=10)
    ax1.text(target[0], target[1], target[2] + 0.04, "Target",
             fontsize=9, ha="center", color="green")
    ax1.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    setup_3d_axes(ax2, "Replanned Path (Avoidance)")
    ax2.plot_surface(sx, sy, sz, color="red", alpha=0.3)
    ax2.text(obs_center[0], obs_center[1], obs_center[2] + obs_radius + 0.03,
             "Obstacle", fontsize=9, ha="center", color="red", fontweight="bold")

    if avoid_traj is not None:
        avoid_ee = np.array([forward_kinematics(*q)[3] for q in avoid_traj])
        ax2.plot(avoid_ee[:, 0], avoid_ee[:, 1], avoid_ee[:, 2],
                 "-", color="seagreen", linewidth=2, alpha=0.8,
                 label="Avoidance path")
        if waypoint is not None:
            ax2.scatter(*waypoint, s=80, c="orange", marker="D", zorder=10)
            ax2.text(waypoint[0], waypoint[1], waypoint[2] + 0.04,
                     "Via-point", fontsize=8, ha="center", color="darkorange")
        draw_arm(ax2, *avoid_traj[-1], color="seagreen", alpha=0.9)
        ax2.legend(loc="upper left", fontsize=8)
    else:
        ax2.text2D(0.5, 0.5, "No avoidance\npath found",
                   transform=ax2.transAxes, fontsize=14,
                   ha="center", va="center", color="red")

    ax2.scatter(*target, s=120, c="limegreen", marker="*", zorder=10)
    ax2.text(target[0], target[1], target[2] + 0.04, "Target",
             fontsize=9, ha="center", color="green")

    plt.suptitle("Obstacle Avoidance — Direct vs Replanned",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/arm_obstacle_avoidance.png", dpi=150, bbox_inches="tight")
    print("Saved: assets/arm_obstacle_avoidance.png")
    if not SAVE_ONLY:
        plt.show()
    plt.close()


def demo_obstacle_gif():
    obs_center = np.array([0.20, 0.10, 0.40])
    obs_radius = 0.06

    start_q = np.array([0.0, 0.0, 0.0])
    target = np.array([0.15, 0.15, 0.45])

    ik_direct = analytical_ik(*target)
    if ik_direct is None:
        print("Target unreachable, skipping obstacle GIF")
        return
    goal_q = np.array(ik_direct)
    direct_traj = smoothstep_trajectory(start_q, goal_q, steps=40)
    collision_idx = check_collision(direct_traj, obs_center, obs_radius)
    if collision_idx < 0:
        collision_idx = len(direct_traj) - 1

    avoid_traj, via_pt = replan_around_obstacle(
        start_q, target, obs_center, obs_radius)
    if avoid_traj is None:
        print("No avoidance path found, skipping GIF")
        return

    direct_frames = direct_traj[:collision_idx + 1]
    flash_frames = [direct_traj[collision_idx]] * 8
    return_traj = smoothstep_trajectory(
        np.array(direct_traj[collision_idx]), start_q, steps=20)
    avoid_frames = avoid_traj
    hold_frames = [avoid_traj[-1]] * 15

    phases = (
        [("direct", q) for q in direct_frames] +
        [("flash", q) for q in flash_frames] +
        [("return", q) for q in return_traj] +
        [("avoid", q) for q in avoid_frames] +
        [("hold", q) for q in hold_frames]
    )

    u_mesh, v_mesh = np.mgrid[0:2*np.pi:20j, 0:np.pi:15j]
    sx = obs_center[0] + obs_radius * np.cos(u_mesh) * np.sin(v_mesh)
    sy = obs_center[1] + obs_radius * np.sin(u_mesh) * np.sin(v_mesh)
    sz = obs_center[2] + obs_radius * np.cos(v_mesh)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    direct_trace = {"x": [], "y": [], "z": []}
    avoid_trace = {"x": [], "y": [], "z": []}

    def update(frame_idx):
        ax.cla()
        setup_3d_axes(ax, "Obstacle Avoidance Animation")

        phase, q = phases[frame_idx]

        ax.plot_surface(sx, sy, sz, color="red", alpha=0.3)
        ax.text(obs_center[0], obs_center[1],
                obs_center[2] + obs_radius + 0.03,
                "Obstacle", fontsize=9, ha="center", color="red",
                fontweight="bold")

        ax.scatter(*target, s=120, c="limegreen", marker="*", zorder=10)
        ax.text(target[0], target[1], target[2] + 0.04, "Target",
                fontsize=9, ha="center", color="green")

        if via_pt is not None:
            ax.scatter(*via_pt, s=60, c="orange", marker="D",
                       zorder=10, alpha=0.7)

        _, _, _, ee = forward_kinematics(*q)

        if phase == "direct":
            direct_trace["x"].append(ee[0])
            direct_trace["y"].append(ee[1])
            direct_trace["z"].append(ee[2])
        elif phase == "avoid":
            avoid_trace["x"].append(ee[0])
            avoid_trace["y"].append(ee[1])
            avoid_trace["z"].append(ee[2])

        if direct_trace["x"]:
            ax.plot(direct_trace["x"], direct_trace["y"],
                    direct_trace["z"],
                    "-", color="crimson", linewidth=1.5, alpha=0.6)

        if avoid_trace["x"]:
            ax.plot(avoid_trace["x"], avoid_trace["y"],
                    avoid_trace["z"],
                    "-", color="seagreen", linewidth=2, alpha=0.8)

        if phase == "direct":
            arm_color = "crimson"
            status = "Direct approach..."
        elif phase == "flash":
            arm_color = "red" if frame_idx % 2 == 0 else "darkred"
            status = "COLLISION DETECTED!"
            ax.scatter(*ee, s=200, c="red", marker="X", zorder=10)
        elif phase == "return":
            arm_color = "gray"
            status = "Replanning..."
        elif phase == "avoid":
            arm_color = "seagreen"
            status = "Following avoidance path"
        else:
            arm_color = "seagreen"
            status = "Target reached!"

        draw_arm(ax, q[0], q[1], q[2], color=arm_color, alpha=0.9)

        ax.text2D(0.02, 0.95, status,
                  transform=ax.transAxes, fontsize=11,
                  verticalalignment="top",
                  bbox=dict(boxstyle="round,pad=0.3",
                            facecolor="lightyellow", alpha=0.9))

    anim = animation.FuncAnimation(
        fig, update, frames=len(phases), interval=60, repeat=False)

    anim.save("assets/arm_obstacle_avoidance.gif", writer="pillow", fps=20)
    print("Saved: assets/arm_obstacle_avoidance.gif")
    if not SAVE_ONLY:
        plt.show()
    plt.close()


def main():
    global SAVE_ONLY
    parser = argparse.ArgumentParser(description="3D Robotic Arm Visualization")
    parser.add_argument("--demo", type=int, choices=[1, 2, 3, 4, 5], default=0,
                        help="1=static, 2=animated, 3=workspace, 4=obstacle, "
                             "5=obstacle GIF. 0=all")
    parser.add_argument("--save-only", action="store_true",
                        help="Save images/GIF without opening display")
    args = parser.parse_args()

    SAVE_ONLY = args.save_only
    _init_matplotlib()

    if args.demo == 0 or args.demo == 1:
        demo_static()
    if args.demo == 0 or args.demo == 2:
        demo_animated()
    if args.demo == 0 or args.demo == 3:
        demo_workspace()
    if args.demo == 0 or args.demo == 4:
        demo_obstacle()
    if args.demo == 0 or args.demo == 5:
        demo_obstacle_gif()


if __name__ == "__main__":
    main()
