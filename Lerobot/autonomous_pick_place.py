"""
SO-101 MuJoCo Pick-and-Place — DYNAMIC DETECT & PICK (v8)

Major upgrade from v6.1:
  - Cube spawns at a RANDOM position on the table each episode
  - Arm SWEEPS shoulder_pan left→right to SEARCH for the cube
  - Once detected (via camera + ground-truth pose), all waypoints
    are solved dynamically via IK for the detected position
  - Pick-place-return cycle, then a NEW random spawn, repeat forever

Architecture:
  SPAWN → SWEEP_SCAN → DETECT → IK_SOLVE → PRE → GRAB → CLOSE → LIFT
  → PAN_TO_PLACE → PLACE → RELEASE → RETURN → (loop)

Reachable workspace (from diagnostic):
  X: [0.10, 0.26], Y: [-0.18, 0.18], Pan: [-55°, 55°]
"""
import os
import time
import math
import random
import numpy as np
import cv2
import mujoco
import mujoco.viewer
import scipy.optimize

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

SCENE_XML_PATH = "/Users/udbhavkulkarni/Downloads/Future-of-robotics-main/physical-ai-challenge-2026/macos_pipeline/sim/robots/SO101/scene.xml"

# Place target is always the green marker
CUBE_PLACE_POS = np.array([0.18, -0.12, 0.065])
CUBE_HALF_SIZE = 0.015  # 3cm cube

# Spawn zone: anywhere on the table the arm can reach
SPAWN_X_RANGE = (0.12, 0.25)
SPAWN_Y_RANGE = (-0.16, 0.16)

# Scan parameters
SCAN_PAN_MIN = math.radians(-55)
SCAN_PAN_MAX = math.radians(55)
SCAN_SPEED = 0.0008  # rad/step — full sweep in ~2.4 seconds


def solve_ik(model, data, site_id, target_pos, pan_hint=0.0):
    """IK solver: find joint config so gripperframe reaches target_pos."""
    def cost(q):
        data.qpos[:5] = q
        mujoco.mj_kinematics(model, data)
        pos_err = np.linalg.norm(data.site_xpos[site_id] - target_pos) * 100
        return pos_err + (q[3] - 1.65)**2 * 2
    res = scipy.optimize.minimize(
        cost, [pan_hint, -0.157, 0.243, 1.651, 0.0], method='SLSQP',
        bounds=[(-1.92, 1.92), (-1.75, 1.75), (-1.69, 1.69),
                (-1.66, 1.66), (-2.74, 2.84)])
    return list(res.x)


def random_spawn_pos():
    """Generate a random cube position on the table within reach."""
    x = random.uniform(*SPAWN_X_RANGE)
    y = random.uniform(*SPAWN_Y_RANGE)
    z = 0.05 + CUBE_HALF_SIZE  # table_top + half cube
    return np.array([x, y, z])


def create_simulation():
    """Build scene with table, cube (random pos), and place marker."""
    print(f"[*] Loading SO-101 from: {SCENE_XML_PATH}")
    with open(SCENE_XML_PATH, "r") as f:
        xml_content = f.read()

    # Initial cube position (will be reset each episode)
    init_pos = random_spawn_pos()
    hs = CUBE_HALF_SIZE
    injections = f"""
        <body name="table" pos="0.25 0 0.025">
            <geom type="box" size="0.2 0.2 0.025" rgba="0.6 0.6 0.6 1"/>
        </body>
        <body name="red_cube" pos="{init_pos[0]} {init_pos[1]} {init_pos[2]}">
            <freejoint/>
            <geom name="cube_geom" type="box" size="{hs} {hs} {hs}"
                  rgba="1 0 0 1" mass="0.05" friction="2.0 0.1 0.001"/>
        </body>
        <body name="place_marker" pos="{CUBE_PLACE_POS[0]} {CUBE_PLACE_POS[1]} {CUBE_PLACE_POS[2]}">
            <geom type="box" size="0.025 0.025 0.003" rgba="0 1 0 0.5"
                  contype="0" conaffinity="0"/>
        </body>
        <camera name="rgbd_cam" pos="0.15 0.5 0.4" fovy="50"
                xyaxes="1 0 0 0 -0.8 0.6"/>
    """
    xml_content = xml_content.replace('</worldbody>', f'    {injections}\n    </worldbody>')
    os.chdir(os.path.dirname(SCENE_XML_PATH))

    model = mujoco.MjModel.from_xml_string(xml_content)
    data = mujoco.MjData(model)
    return model, data


def respawn_cube(data, cube_qpos_adr):
    """Teleport cube to a new random position on the table."""
    new_pos = random_spawn_pos()
    data.qpos[cube_qpos_adr:cube_qpos_adr + 3] = new_pos
    data.qpos[cube_qpos_adr + 3:cube_qpos_adr + 7] = [1, 0, 0, 0]  # upright
    data.qvel[:] = 0  # zero everything
    return new_pos


def main():
    print("=" * 62)
    print("🤖  PHYSICAL AI HACKATHON 2026 — DYNAMIC DETECT & PICK v8")
    print("=" * 62)

    yolo_model = None
    if HAS_YOLO:
        print("[*] Loading YOLOv8...")
        try:
            yolo_model = YOLO("yolov8n.pt")
        except Exception as e:
            print(f"[!] YOLOv8 load failed: {e}")

    model, data = create_simulation()
    renderer = mujoco.Renderer(model, 480, 640)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_cube")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    cube_jnt = model.body_jntadr[cube_id]
    cube_qpos = model.jnt_qposadr[cube_jnt]
    cube_dof = model.jnt_dofadr[cube_jnt]

    # ─── STATE MACHINE ───
    PHASE_SPAWN = "SPAWN"
    PHASE_SCAN = "SCAN"
    PHASE_WAYPOINTS = "WAYPOINTS"
    PHASE_DONE = "EPISODE_DONE"

    phase = PHASE_SPAWN
    episode = 0
    scan_pan = SCAN_PAN_MIN
    scan_dir = 1  # +1 = sweep right, -1 = sweep left
    detected_pos = None
    waypoints = []
    wp_idx = 0
    step_in_wp = 0

    # Neutral scan pose: arm up, gripper open, ready to sweep
    SCAN_JOINTS = [0.0, -0.5, 0.3, 1.0, 0.0, 1.5]  # arm slightly raised for scanning

    print("[✓] Ready. Run with: mjpython autonomous_pick_place.py\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            step_count = 0
            while viewer.is_running():
                step_start = time.time()

                # ─── VISION (every 25 steps) ───
                if step_count % 25 == 0:
                    renderer.update_scene(data, camera="rgbd_cam")
                    img = renderer.render()
                    cv2_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                    # YOLO detection overlay
                    if yolo_model is not None:
                        results = yolo_model(cv2_img, verbose=False)
                        if len(results) > 0 and len(results[0].boxes) > 0:
                            for box in results[0].boxes:
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                cv2.rectangle(cv2_img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                                cv2.putText(cv2_img, f"YOLOv8 ({box.conf[0]:.2f})",
                                            (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                            0.4, (0, 255, 255), 1)

                    # HUD overlays
                    cube_pos = data.xpos[cube_id]
                    grip_pos = data.site_xpos[site_id]
                    cv2.putText(cv2_img,
                                f"Cube: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]",
                                (10, 465), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    cv2.putText(cv2_img,
                                f"Grip: [{grip_pos[0]:.3f}, {grip_pos[1]:.3f}, {grip_pos[2]:.3f}]",
                                (10, 448), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
                    cv2.putText(cv2_img,
                                f"Phase: {phase} | Episode: {episode}",
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    if phase == PHASE_SCAN:
                        cv2.putText(cv2_img,
                                    f"Pan: {math.degrees(scan_pan):.1f} deg",
                                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # ═══════════════════════════════════════
                # ─── PHASE: SPAWN ───
                # ═══════════════════════════════════════
                if phase == PHASE_SPAWN:
                    episode += 1
                    # Reset arm to neutral
                    data.qpos[:6] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    data.ctrl[:6] = [0.0, 0.0, 0.0, 0.0, 0.0, 1.5]
                    # Respawn cube at random location
                    new_pos = respawn_cube(data, cube_qpos)
                    # Re-enable collision
                    model.geom_contype[cube_geom_id] = 1
                    model.geom_conaffinity[cube_geom_id] = 1

                    print(f"\n{'='*62}")
                    print(f"🎲  EPISODE {episode}: Cube spawned at [{new_pos[0]:.3f}, {new_pos[1]:.3f}]")
                    print(f"{'='*62}")

                    # Initialize scan
                    scan_pan = SCAN_PAN_MIN
                    scan_dir = 1
                    detected_pos = None
                    phase = PHASE_SCAN
                    step_in_wp = 0
                    print(f"[🔍] Scanning for cube...")

                # ═══════════════════════════════════════
                # ─── PHASE: SCAN (sweep shoulder_pan) ──
                # ═══════════════════════════════════════
                elif phase == PHASE_SCAN:
                    # Set arm to scan pose with current pan angle
                    data.ctrl[0] = scan_pan
                    data.ctrl[1:6] = SCAN_JOINTS[1:]

                    # Directly set pan qpos for smooth sweep (bypass weak servo)
                    data.qpos[0] = scan_pan
                    data.qvel[0] = 0.0

                    # Advance pan angle
                    scan_pan += SCAN_SPEED * scan_dir
                    if scan_pan >= SCAN_PAN_MAX:
                        scan_pan = SCAN_PAN_MAX
                        scan_dir = -1
                    elif scan_pan <= SCAN_PAN_MIN:
                        scan_pan = SCAN_PAN_MIN
                        scan_dir = 1

                    # Check detection: is the cube within the arm's reach arc?
                    # Use ground-truth pose (DenseFusion stand-in)
                    cube_pos = data.xpos[cube_id].copy()
                    cube_pan = math.atan2(cube_pos[1], cube_pos[0])

                    # "Detect" when the scan angle is close to the cube's angle
                    pan_diff = abs(scan_pan - cube_pan)
                    if pan_diff < math.radians(8):  # within 8° of cube direction
                        detected_pos = cube_pos.copy()
                        # Adjust Z to table height
                        detected_pos[2] = 0.05 + CUBE_HALF_SIZE

                        det_pan_deg = math.degrees(cube_pan)
                        print(f"[✓] DETECTED cube at [{detected_pos[0]:.3f}, {detected_pos[1]:.3f}] "
                              f"(pan={det_pan_deg:.1f}°)")

                        # ─── SOLVE IK FOR DETECTED POSITION ───
                        print(f"[*] Computing dynamic IK for detected position...")
                        pan_hint = cube_pan

                        GRAB_J = solve_ik(model, data, site_id, detected_pos, pan_hint)
                        PRE_target = detected_pos.copy()
                        PRE_target[0] -= 0.04 * math.cos(cube_pan)
                        PRE_target[1] -= 0.04 * math.sin(cube_pan)
                        PRE_J = solve_ik(model, data, site_id, PRE_target, pan_hint)

                        LIFT_target = detected_pos.copy()
                        LIFT_target[2] = 0.18
                        LIFT_J = solve_ik(model, data, site_id, LIFT_target, pan_hint)

                        PLACE_J = solve_ik(model, data, site_id, CUBE_PLACE_POS, 0.8)
                        PLACE_LIFT_target = CUBE_PLACE_POS.copy()
                        PLACE_LIFT_target[2] = 0.18
                        PLACE_LIFT_J = solve_ik(model, data, site_id, PLACE_LIFT_target, 0.8)
                        PLACE_PAN = PLACE_J[0]
                        GRAB_PAN = GRAB_J[0]

                        print(f"    GRAB pan={math.degrees(GRAB_PAN):.1f}° "
                              f"joints={[f'{j:.3f}' for j in GRAB_J]}")
                        print(f"    PLACE pan={math.degrees(PLACE_PAN):.1f}°")

                        # Build dynamic waypoint sequence
                        waypoints = [
                            # First: pan to the cube's direction (from current scan position)
                            ("PAN_TO_CUBE",
                             [GRAB_PAN] + SCAN_JOINTS[1:],
                             False, 800, 1500,
                             (scan_pan, GRAB_PAN, 800)),

                            ("PRE_APPROACH",
                             PRE_J + [1.5],
                             False, 750, 2500, None),

                            ("GRAB_ADVANCE",
                             GRAB_J + [1.5],
                             False, 1000, 2500, None),

                            ("CLOSE_GRIP",
                             GRAB_J + [-0.10],
                             True, 1000, 2000, None),

                            ("LIFT",
                             LIFT_J + [-0.10],
                             True, 750, 2500, None),

                            ("PAN_TO_PLACE",
                             [PLACE_PAN] + LIFT_J[1:] + [-0.10],
                             True, 1500, 2000,
                             (GRAB_PAN, PLACE_PAN, 1500)),

                            ("PLACE_LIFT",
                             PLACE_LIFT_J + [-0.10],
                             True, 1000, 2500, None),

                            ("PLACE_LOWER",
                             PLACE_J + [-0.10],
                             True, 1000, 2500, None),

                            ("RELEASE",
                             PLACE_J + [1.5],
                             False, 750, 1500, None),

                            ("PAN_BACK",
                             [0.0] + SCAN_JOINTS[1:],
                             False, 1200, 2000,
                             (PLACE_PAN, 0.0, 1200)),

                            ("RETURN",
                             [0.0, 0.0, 0.0, 0.0, 0.0, 1.5],
                             False, 750, 1500, None),
                        ]

                        phase = PHASE_WAYPOINTS
                        wp_idx = 0
                        step_in_wp = 0
                        print(f"[→] Phase 1/{len(waypoints)}: {waypoints[0][0]}")

                # ═══════════════════════════════════════
                # ─── PHASE: EXECUTE WAYPOINTS ──────────
                # ═══════════════════════════════════════
                elif phase == PHASE_WAYPOINTS and wp_idx < len(waypoints):
                    name, ctrl, hold_cube, min_steps, max_steps, pan_interp = waypoints[wp_idx]
                    target = np.array(ctrl)
                    data.ctrl[:6] = ctrl

                    # Smooth pan interpolation (bypass weak servo)
                    if pan_interp is not None:
                        s_pan, e_pan, n_steps = pan_interp
                        t = min(step_in_wp / n_steps, 1.0)
                        t = t * t * (3.0 - 2.0 * t)  # hermite ease
                        data.qpos[0] = s_pan + t * (e_pan - s_pan)
                        data.qvel[0] = 0.0
                        data.ctrl[0] = data.qpos[0]

                    # Magnetic grasp with collision toggle
                    if hold_cube:
                        model.geom_contype[cube_geom_id] = 0
                        model.geom_conaffinity[cube_geom_id] = 0
                        data.qpos[cube_qpos:cube_qpos + 3] = data.site_xpos[site_id]
                        data.qpos[cube_qpos + 3:cube_qpos + 7] = [1, 0, 0, 0]
                        data.qvel[cube_dof:cube_dof + 6] = 0
                    else:
                        model.geom_contype[cube_geom_id] = 1
                        model.geom_conaffinity[cube_geom_id] = 1

                    step_in_wp += 1

                    # Convergence check
                    if pan_interp is not None:
                        max_err = np.max(np.abs(data.qpos[1:6] - target[1:]))
                    else:
                        max_err = np.max(np.abs(data.qpos[:6] - target))

                    if (max_err < 0.03 and step_in_wp >= min_steps) or step_in_wp >= max_steps:
                        wp_idx += 1
                        step_in_wp = 0
                        if wp_idx < len(waypoints):
                            print(f"[→] Phase {wp_idx+1}/{len(waypoints)}: {waypoints[wp_idx][0]}")
                        else:
                            # Episode complete
                            fc = data.xpos[cube_id]
                            err = np.linalg.norm(fc[:2] - CUBE_PLACE_POS[:2])
                            print(f"[✓] ══════ EPISODE {episode} COMPLETE ══════")
                            print(f"    Final cube: [{fc[0]:.4f}, {fc[1]:.4f}, {fc[2]:.4f}]")
                            print(f"    Target:     [{CUBE_PLACE_POS[0]:.4f}, {CUBE_PLACE_POS[1]:.4f}]")
                            print(f"    XY error:   {err*1000:.1f} mm")
                            phase = PHASE_DONE
                            step_in_wp = 0

                # ═══════════════════════════════════════
                # ─── PHASE: DONE → RESPAWN ─────────────
                # ═══════════════════════════════════════
                elif phase == PHASE_DONE:
                    step_in_wp += 1
                    if step_in_wp >= 1500:  # 3 second pause, then respawn
                        phase = PHASE_SPAWN

                # ─── STEP ───
                mujoco.mj_step(model, data)
                viewer.sync()
                step_count += 1

                dt = model.opt.timestep - (time.time() - step_start)
                if dt > 0:
                    time.sleep(dt)

    except RuntimeError:
        print("\n[!] FATAL: launch_passive requires mjpython on macOS.")
        print("    Run:  mjpython autonomous_pick_place.py")


if __name__ == "__main__":
    main()
