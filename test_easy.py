import gymnasium as gym
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv # <--- Re-added this import
import pygame
import numpy as np
import math
import argparse
import os
import sys
import json
import datetime

# --- RL IMPORTS ---
try:
    from stable_baselines3 import PPO, SAC
except ImportError:
    print("Error: Stable Baselines3 not found.")
    print("Please run: pip install stable-baselines3 shimmy")
    sys.exit(1)

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="Universal Drone System: Train, Fly, Record")

# RL & Mode Args
parser.add_argument("--train", action="store_true", help="Training Mode (No Graphics)")
parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO", help="RL Algorithm")
parser.add_argument("--steps", type=int, default=100000, help="Training timesteps")
parser.add_argument("--pilot", type=str, choices=["human", "agent"], default="human", help="Who flies?")
parser.add_argument("--model_path", type=str, default="fixedwing_agent", help="Agent file path (no ext)")
parser.add_argument("--render-mode", type=str, choices=["human", "none"], default="human", help="Render Mode")

# Visual & Sim Args
parser.add_argument("--zone", type=float, default=0.0, help="Zone Radius (0 = Auto)")
parser.add_argument("--disable-hud", action="store_true", help="Disable ALL HUD")
parser.add_argument("--no-horizon", action="store_true", help="Disable Artificial Horizon")
parser.add_argument("--show-data", action="store_true", help="Show text telemetry")
args = parser.parse_args()

# --- CONFIG ---
AXIS_ROLL, AXIS_PITCH, AXIS_YAW, AXIS_THROTTLE = 0, 1, 2, 3
BTN_PAUSE, BTN_PIP, BTN_RADAR, BTN_RESET = 1, 2, 3, 7 
INVERT_PITCH, INVERT_THROTTLE = True, True

EXPO_VALUE = 0.6  
MAX_ROLL_RATE = 0.5
MAX_PITCH_RATE = 0.5
MAX_YAW_RATE = 0.3

# --- DATA RECORDING SETUP ---
output_dir = "flight_data"
os.makedirs(output_dir, exist_ok=True)
data_buffer = {"observations": [], "actions": [], "rewards": [], "terminals": []}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# --- 1. TRAINING MODE ---
if args.train:
    print(f"\n=== TRAINING MODE ===")
    print(f"Algorithm: {args.algo}")
    print(f"Steps: {args.steps}")
    print(f"Output: {args.model_path}.zip")
    
    # 1. Create Headless Environment
    try:
        env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode=None)
    except:
        env = gym.make("PyFlyt/Fixedwing-Waypoints-v0", render_mode=None)
    
    # 2. FLATTEN THE ENVIRONMENT (Matches your train.py)
    env = FlattenWaypointEnv(env, context_length=2)

    # 3. Initialize & Train (MlpPolicy for flat envs)
    ModelClass = SAC if args.algo == "SAC" else PPO
    model = ModelClass("MlpPolicy", env, verbose=1)
    
    model.learn(total_timesteps=args.steps)
    model.save(args.model_path)
    
    print(f"\nTraining Complete! Saved to {args.model_path}.zip")
    env.close()
    sys.exit()

# --- 2. FLIGHT MODE ---

# Initialize Environment with Graphics
render_mode = args.render_mode if args.render_mode != "none" else None
try:
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode=render_mode)
except:
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v0", render_mode=render_mode)

# === CRITICAL FIX: APPLY SAME WRAPPER AS TRAINING ===
env = FlattenWaypointEnv(env, context_length=2)
# ====================================================

# Load Agent
agent_model = None
if args.pilot == "agent":
    path = f"{args.model_path}.zip"
    if not os.path.exists(path):
        print(f"Error: Agent file '{path}' not found.")
        print("Train one first using --train")
        sys.exit()
    
    print(f"Loading {args.algo} Agent from {path}...")
    
    # Select correct class for loading
    ModelClass = SAC if args.algo == "SAC" else PPO
    agent_model = ModelClass.load(path)
    print("Agent Loaded. AI has control.")

# Detect Zone
ZONE_RADIUS = args.zone
if ZONE_RADIUS == 0.0:
    try: ZONE_RADIUS = env.unwrapped.env.flight_dome_size
    except: ZONE_RADIUS = 100.0

# Physics Client
try:
    if hasattr(env.unwrapped, 'ctx'): p = env.unwrapped.ctx.pybullet_client
    elif hasattr(env.unwrapped, 'env'): p = env.unwrapped.env.aviary.ctx.pybullet_client
    else: import pybullet as p
except: import pybullet as p

# Pygame Setup
pygame.init()
pygame.joystick.init()
MAIN_RENDER_W, MAIN_RENDER_H = 960, 540
WINDOW_W, WINDOW_H = 1920, 1080 
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption(f"Pilot: {args.pilot.upper()} | Algo: {args.algo}")

# PIP Settings
PIP_SIZE = (240, 180)
PIP_POS = (WINDOW_W - 250, 10)

font = pygame.font.SysFont("monospace", 20, bold=True)
warning_font = pygame.font.SysFont("monospace", 40, bold=True)
capture_font = pygame.font.SysFont("monospace", 60, bold=True)
small_font = pygame.font.SysFont("monospace", 16)

joystick = None
if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

# --- HELPER FUNCTIONS ---
def get_drone_state(env):
    try:
        drone = env.unwrapped.env.drones[0]
        pos, orn = p.getBasePositionAndOrientation(drone.Id)
        euler = p.getEulerFromQuaternion(orn)
        return drone.Id, pos, orn, euler
    except: return None, None, None, None

def save_data():
    if len(data_buffer["observations"]) == 0: return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    npz_filename = os.path.join(output_dir, f"log_{args.pilot}_{timestamp}.npz")
    print(f"\nSaving {len(data_buffer['observations'])} steps...")
    
    np.savez_compressed(
        npz_filename,
        obs=np.array(data_buffer["observations"], dtype=object),
        actions=np.array(data_buffer["actions"]),
        rewards=np.array(data_buffer["rewards"]),
        dones=np.array(data_buffer["terminals"])
    )
    print("Save Complete!")

def draw_zone_boundary(radius):
    try:
        p.removeAllUserDebugItems()
        num_segments = 24 
        heights = [0, 20] 
        for h in heights:
            for i in range(num_segments):
                angle1 = (2 * math.pi * i) / num_segments
                angle2 = (2 * math.pi * (i + 1)) / num_segments
                x1, y1 = radius * math.cos(angle1), radius * math.sin(angle1)
                x2, y2 = radius * math.cos(angle2), radius * math.sin(angle2)
                p.addUserDebugLine([x1, y1, h], [x2, y2, h], [1, 0, 0], lineWidth=3)
        for i in range(8):
            angle = (2 * math.pi * i) / 8
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            p.addUserDebugLine([x, y, 0], [x, y, 20], [1, 0, 0], lineWidth=2)
    except: pass

def render_camera(drone_id, pos, orn, mode="chase", w=320, h=240):
    if drone_id is None: return None
    rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    cam_offset = [0.5, 0, 0.1] if mode == "fpv" else [-3.5, 0, 1.0] 
    target_offset = [5.0, 0, 0.0] if mode == "fpv" else [0, 0, 0]
    fov = 80 if mode == "fpv" else 60
    cam_pos = np.array(pos) + rot_mat.dot(cam_offset)
    cam_target = np.array(pos) + rot_mat.dot(target_offset)
    view_matrix = p.computeViewMatrix(cam_pos, cam_target, rot_mat.dot([0, 0, 1]))
    proj_matrix = p.computeProjectionMatrixFOV(fov, float(w)/h, 0.1, 1000.0)
    _, _, rgb, _, _ = p.getCameraImage(width=w, height=h, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.array(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
    return pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))

def draw_arrow(surface, color, center, angle_rad, size=10):
    points = [(size, 0), (-size*0.6, -size*0.6), (-size*0.6, size*0.6)]
    rotated = []
    for x, y in points:
        rx = x * math.cos(angle_rad) - y * math.sin(angle_rad)
        ry = x * math.sin(angle_rad) + y * math.cos(angle_rad)
        rotated.append((center[0] + rx, center[1] - ry))
    pygame.draw.polygon(surface, color, rotated)

def draw_radar(screen, drone_pos, drone_yaw, targets, zone_radius, mode):
    RADAR_SIZE, CENTER = 200, (WINDOW_W - 120, WINDOW_H - 120)
    PX_PER_METER = (RADAR_SIZE / 2) / (zone_radius * 1.5)
    
    s = pygame.Surface((RADAR_SIZE, RADAR_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(s, (0, 40, 0, 200), (RADAR_SIZE//2, RADAR_SIZE//2), RADAR_SIZE // 2)
    screen.blit(s, (CENTER[0] - RADAR_SIZE//2, CENTER[1] - RADAR_SIZE//2))
    pygame.draw.circle(screen, (150, 150, 150), CENTER, RADAR_SIZE // 2, 2)
    
    rot_offset = (math.pi / 2) - drone_yaw if mode == 1 else 0

    def world_to_radar(wx, wy):
        if mode == 1: wx, wy = wx - drone_pos[0], wy - drone_pos[1]
        sx, sy = wx * PX_PER_METER, wy * PX_PER_METER
        if mode == 1:
            rx = sx * math.cos(rot_offset) - sy * math.sin(rot_offset)
            ry = sx * math.sin(rot_offset) + sy * math.cos(rot_offset)
            sx, sy = rx, ry
        return int(CENTER[0] + sx), int(CENTER[1] - sy)

    zx, zy = world_to_radar(0, 0)
    pygame.draw.circle(screen, (255, 50, 50), (zx, zy), int(zone_radius * PX_PER_METER), 3)

    if targets is not None:
        for delta in targets:
            tx, ty = drone_pos[0] + delta[0], drone_pos[1] + delta[1]
            rx, ry = world_to_radar(tx, ty)
            vec_x, vec_y = rx - CENTER[0], ry - CENTER[1]
            if math.sqrt(vec_x**2 + vec_y**2) > (RADAR_SIZE // 2) - 6:
                ratio = ((RADAR_SIZE // 2) - 6) / math.sqrt(vec_x**2 + vec_y**2)
                rx, ry = int(CENTER[0] + vec_x * ratio), int(CENTER[1] + vec_y * ratio)
            pygame.draw.circle(screen, (255, 255, 0), (rx, ry), 5)

    if mode == 1:
        draw_arrow(screen, (255, 255, 255), CENTER, math.pi/2, size=12)
    else:
        dx, dy = world_to_radar(drone_pos[0], drone_pos[1])
        vec_x, vec_y = dx - CENTER[0], dy - CENTER[1]
        if math.sqrt(vec_x**2 + vec_y**2) > (RADAR_SIZE // 2) - 8:
             ratio = ((RADAR_SIZE // 2) - 8) / math.sqrt(vec_x**2 + vec_y**2)
             dx, dy = int(CENTER[0] + vec_x * ratio), int(CENTER[1] + vec_y * ratio)
        draw_arrow(screen, (255, 255, 255), (dx, dy), drone_yaw, size=12)

    font_radar = pygame.font.SysFont("monospace", 12, bold=True)
    lbl = font_radar.render("HEAD UP" if mode == 1 else "NORTH UP", True, (200, 200, 200))
    screen.blit(lbl, (CENTER[0] - 30, CENTER[1] + RADAR_SIZE//2 + 5))

# --- MAIN LOOP ---
clock = pygame.time.Clock()
obs, _ = env.reset()
draw_zone_boundary(ZONE_RADIUS)

action = np.array([0.0, 0.0, 0.0, 0.0])
current_targets = []

# Capture Message State
capture_timer = 0
last_target_count = 0

paused = True 
running = True
show_pip = True
radar_mode = 0 # 0 = North Up
TARGET_FPS = 30
LOG_FREQUENCY = 10

try:
    while running:
        clock.tick(TARGET_FPS)
        drone_id, pos, orn, euler = get_drone_state(env)
        pygame.event.pump() 

        if not paused:
            if args.pilot == "human" and joystick:
                raw_roll = joystick.get_axis(AXIS_ROLL)
                raw_pitch = joystick.get_axis(AXIS_PITCH)
                if INVERT_PITCH: raw_pitch = -raw_pitch
                raw_yaw = joystick.get_axis(AXIS_YAW)
                raw_thr = joystick.get_axis(AXIS_THROTTLE)
                if INVERT_THROTTLE: thrust_cmd = (raw_thr * -1.0 + 1.0) / 2.0
                else: thrust_cmd = (raw_thr + 1.0) / 2.0

                def process_stick(val, expo, rate):
                    if abs(val) < 0.05: val = 0.0
                    val = (val**3 * expo) + (val * (1 - expo))
                    return val * rate
                
                action[0] = process_stick(raw_roll, EXPO_VALUE, MAX_ROLL_RATE)
                action[1] = process_stick(raw_pitch, EXPO_VALUE, MAX_PITCH_RATE)
                action[2] = process_stick(raw_yaw, EXPO_VALUE, MAX_YAW_RATE)
                action[3] = np.clip(thrust_cmd, 0.0, 1.0)
                
            elif args.pilot == "agent":
                # Agent sees flattened obs
                action, _states = agent_model.predict(obs, deterministic=True)

            # Record
            data_buffer["observations"].append(obs)
            data_buffer["actions"].append(action.copy()) 

            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            data_buffer["rewards"].append(reward)
            data_buffer["terminals"].append(terminated or truncated)

            # --- TARGET EXTRACTION ---
            # Direct Access to internal Waypoints object
            if hasattr(env.unwrapped, "waypoints") and hasattr(env.unwrapped.waypoints, "targets"):
                abs_targets = env.unwrapped.waypoints.targets
                current_targets = [t - pos for t in abs_targets]
            else:
                current_targets = []

            # --- CAPTURE DETECTION ---
            if len(current_targets) < last_target_count and last_target_count > 0:
                capture_timer = 45 
                print(">>> TARGET CAPTURED!")
            
            last_target_count = len(current_targets)

            if len(data_buffer["observations"]) % LOG_FREQUENCY == 0:
                print(f"REC {len(data_buffer['observations']):04d} | [RADAR] Targets: {len(current_targets)}")
                print(f"REC {len(data_buffer['observations']):04d} | [RADAR] Targets Reached: {info.get('num_targets_reached', 0)}")

            if terminated or truncated:
                print(">>> Mission Ended. Resetting...")
                obs, _ = env.reset()
                draw_zone_boundary(ZONE_RADIUS)
                action = np.array([0.0, 0.0, 0.0, 0.0])
                current_targets = []
                last_target_count = 0
                paused = True

        # --- RENDER ---
        screen.fill((0, 0, 0))
        main_surf = render_camera(drone_id, pos, orn, mode="chase", w=MAIN_RENDER_W, h=MAIN_RENDER_H)
        if main_surf:
            scaled_surf = pygame.transform.scale(main_surf, (WINDOW_W, WINDOW_H))
            screen.blit(scaled_surf, (0, 0))

        if not args.disable_hud:
            # PIP
            if show_pip and not paused:
                pip_surf = render_camera(drone_id, pos, orn, mode="fpv", w=PIP_SIZE[0], h=PIP_SIZE[1])
                if pip_surf:
                    pygame.draw.rect(screen, (255, 255, 255), (PIP_POS[0]-2, PIP_POS[1]-2, PIP_SIZE[0]+4, PIP_SIZE[1]+4), 2)
                    screen.blit(pip_surf, PIP_POS)

            # Horizon
            if euler and not args.no_horizon:
                roll_rad, pitch_rad = euler[0], euler[1]
                cx, cy = WINDOW_W // 2, WINDOW_H // 2
                x1 = cx - 300 * math.cos(-roll_rad)
                y1 = (cy + int(math.degrees(pitch_rad) * 6)) - 300 * math.sin(-roll_rad)
                x2 = cx + 300 * math.cos(-roll_rad)
                y2 = (cy + int(math.degrees(pitch_rad) * 6)) + 300 * math.sin(-roll_rad)
                pygame.draw.line(screen, (0, 255, 255), (x1, y1), (x2, y2), 4)
                pygame.draw.circle(screen, (255, 0, 0), (cx, cy), 8) 

            # Zone Warning
            if pos:
                dist = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                if dist > ZONE_RADIUS:
                    if (pygame.time.get_ticks() // 200) % 2 == 0: 
                        warn_surf = warning_font.render(f"OUT OF ZONE!", True, (255, 0, 0))
                        rect = warn_surf.get_rect(center=(WINDOW_W // 2, 100))
                        screen.blit(warn_surf, rect)
                
                # Radar
                draw_radar(screen, pos, euler[2], current_targets, ZONE_RADIUS, radar_mode)

                # Text Data
                if args.show_data:
                    BOX_X, BOX_Y = 20, 80
                    s = pygame.Surface((300, 140))
                    s.set_alpha(150); s.fill((0, 0, 0))
                    screen.blit(s, (BOX_X, BOX_Y))
                    pygame.draw.rect(screen, (255, 255, 255), (BOX_X, BOX_Y, 300, 140), 2)
                    lines = [
                        f"PILOT: {args.pilot.upper()}",
                        f"ALGO:  {args.algo}",
                        f"ALT:   {pos[2]:.1f} m",
                        f"DIST:  {dist:.1f} / {ZONE_RADIUS:.0f}m",
                        f"THR:   {action[3]*100:.0f}%"
                    ]
                    for i, line in enumerate(lines):
                        color = (255, 50, 50) if (i == 3 and dist > ZONE_RADIUS * 0.9) else (0, 255, 0)
                        txt = small_font.render(line, True, color)
                        screen.blit(txt, (BOX_X + 20, BOX_Y + 15 + (i * 24)))

        # Capture Message
        if capture_timer > 0:
            msg = capture_font.render("TARGET CAPTURED!", True, (0, 255, 0))
            rect = msg.get_rect(center=(WINDOW_W // 2, WINDOW_H // 3))
            screen.blit(msg, rect)
            capture_timer -= 1
        
        # Throttle Bar
        throt_h = int(action[3] * 300)
        pygame.draw.rect(screen, (50, 50, 50), (20, WINDOW_H - 350, 40, 300)) 
        pygame.draw.rect(screen, (255, 0, 0), (20, WINDOW_H - 50 - throt_h, 40, throt_h))

        if paused:
            text = font.render(f"PAUSED - {args.pilot.upper()} PILOT", True, (255, 255, 0))
            screen.blit(text, (WINDOW_W//2 - 150, WINDOW_H//2))
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE: paused = not paused
            
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == BTN_PAUSE: paused = not paused
                if event.button == BTN_RADAR: radar_mode = 1 - radar_mode
                if event.button == BTN_PIP: show_pip = not show_pip
                if event.button == BTN_RESET:
                    obs, _ = env.reset()
                    draw_zone_boundary(ZONE_RADIUS)
                    action = np.array([0.0, 0.0, 0.0, 0.0])
                    current_targets = []
                    last_target_count = 0
                    paused = True

except KeyboardInterrupt:
    print("Interrupted...")
finally:
    env.close()
    pygame.quit()
    save_data()