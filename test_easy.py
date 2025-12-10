import gymnasium as gym
import PyFlyt.gym_envs
import pygame
import numpy as np
import math
import datetime
import os
import json
import argparse

# --- ARGUMENTS ---
parser = argparse.ArgumentParser()
parser.add_argument("--disable-hud", action="store_true", help="Disable ALL HUD")
parser.add_argument("--no-horizon", action="store_true", help="Disable Horizon")
parser.add_argument("--show-data", action="store_true", help="Show text data")
parser.add_argument("--zone", type=float, default=0.0, help="Zone Radius (Leave 0 to auto-detect)")
args = parser.parse_args()

# --- CONFIG ---
AXIS_ROLL, AXIS_PITCH, AXIS_YAW, AXIS_THROTTLE = 0, 1, 2, 3
BTN_PAUSE, BTN_PIP, BTN_RADAR, BTN_RESET = 1, 2, 3, 7 
INVERT_PITCH, INVERT_THROTTLE = True, True

# --- FLIGHT TUNING ---
EXPO_VALUE = 0.6  
MAX_ROLL_RATE = 0.5
MAX_PITCH_RATE = 0.5
MAX_YAW_RATE = 0.3

# --- SCREEN SETTINGS (FULL HD) ---
# We render at exactly 50% scale (960x540) to keep FPS high on Intel Graphics
# This scales up perfectly to 1920x1080 without distortion.
MAIN_RENDER_W, MAIN_RENDER_H = 960, 540 
WINDOW_W, WINDOW_H = 1920, 1080
TARGET_FPS = 30
LOG_FREQUENCY = 10

SHOW_PIP = True
RADAR_MODE = 0 # 0 = North Up, 1 = Head Up
PIP_SIZE = (320, 240) # Slightly larger PIP for big screen
PIP_POS = (WINDOW_W - 340, 20)

output_dir = "flight_data"
os.makedirs(output_dir, exist_ok=True)

data_buffer = {"observations": [], "actions": [], "rewards": [], "terminals": []}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def apply_expo(x, expo):
    return (x**3 * expo) + (x * (1 - expo))

# --- ENV SETUP ---
try:
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode="human")
except:
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v0", render_mode="human")

# Auto-detect Zone Size
ZONE_RADIUS = args.zone
if ZONE_RADIUS == 0.0:
    try:
        ZONE_RADIUS = env.unwrapped.env.flight_dome_size
        print(f"Auto-detected Flight Zone: {ZONE_RADIUS} meters")
    except:
        ZONE_RADIUS = 100.0 # Fallback
        print(f"Could not detect zone size. Defaulting to {ZONE_RADIUS} meters")

try:
    if hasattr(env.unwrapped, 'ctx'):
        p = env.unwrapped.ctx.pybullet_client
    elif hasattr(env.unwrapped, 'env'):
        p = env.unwrapped.env.aviary.ctx.pybullet_client
    else:
        import pybullet as p
except:
    import pybullet as p

def get_drone_state(env):
    try:
        drone = env.unwrapped.env.drones[0]
        pos, orn = p.getBasePositionAndOrientation(drone.Id)
        euler = p.getEulerFromQuaternion(orn)
        return drone.Id, pos, orn, euler
    except:
        return None, None, None, None

def draw_zone_boundary(radius):
    """Draws a lighter fence (fewer lines) to prevent buffer overflow."""
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
    except Exception as e:
        print(f"Warning: Could not draw boundary: {e}")

def render_camera(drone_id, pos, orn, mode="chase", w=320, h=240):
    if drone_id is None: return None
    
    rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    
    if mode == "fpv":
        cam_offset = [0.5, 0, 0.1]
        target_offset = [5.0, 0, 0.0]
        fov = 80 
    else: 
        cam_offset = [-3.5, 0, 1.0] 
        target_offset = [0, 0, 0]
        fov = 60

    cam_pos = np.array(pos) + rot_mat.dot(cam_offset)
    cam_target = np.array(pos) + rot_mat.dot(target_offset)
    
    view_matrix = p.computeViewMatrix(cam_pos, cam_target, rot_mat.dot([0, 0, 1]))
    proj_matrix = p.computeProjectionMatrixFOV(fov, float(w)/h, 0.1, 1000.0)

    _, _, rgb, _, _ = p.getCameraImage(
        width=w, height=h,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL 
    )
    
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
    RADAR_SIZE = 200 # Larger radar for big screen
    CENTER = (WINDOW_W - 120, WINDOW_H - 120)
    VIEW_RADIUS_M = zone_radius * 1.2
    PX_PER_METER = (RADAR_SIZE / 2) / VIEW_RADIUS_M
    
    # Background
    s = pygame.Surface((RADAR_SIZE, RADAR_SIZE), pygame.SRCALPHA)
    pygame.draw.circle(s, (0, 40, 0, 200), (RADAR_SIZE//2, RADAR_SIZE//2), RADAR_SIZE // 2)
    screen.blit(s, (CENTER[0] - RADAR_SIZE//2, CENTER[1] - RADAR_SIZE//2))
    pygame.draw.circle(screen, (150, 150, 150), CENTER, RADAR_SIZE // 2, 2)
    
    rotation_offset = 0
    if mode == 1: 
        rotation_offset = (math.pi / 2) - drone_yaw

    def world_to_radar(wx, wy):
        if mode == 1:
            wx -= drone_pos[0]
            wy -= drone_pos[1]
        
        sx = wx * PX_PER_METER
        sy = wy * PX_PER_METER 
        
        if mode == 1:
            rx = sx * math.cos(rotation_offset) - sy * math.sin(rotation_offset)
            ry = sx * math.sin(rotation_offset) + sy * math.cos(rotation_offset)
            sx, sy = rx, ry
        
        return int(CENTER[0] + sx), int(CENTER[1] - sy)

    # Zone
    zx, zy = world_to_radar(0, 0)
    zone_px = int(zone_radius * PX_PER_METER)
    pygame.draw.circle(screen, (255, 50, 50), (zx, zy), zone_px, 3)

    # Targets
    if targets is not None and len(targets) > 0:
        for delta in targets:
            tx = drone_pos[0] + delta[0]
            ty = drone_pos[1] + delta[1]
            rx, ry = world_to_radar(tx, ty)
            
            vec_x = rx - CENTER[0]
            vec_y = ry - CENTER[1]
            dist = math.sqrt(vec_x**2 + vec_y**2)
            max_r = (RADAR_SIZE // 2) - 6
            
            if dist > max_r:
                ratio = max_r / dist
                rx = int(CENTER[0] + vec_x * ratio)
                ry = int(CENTER[1] + vec_y * ratio)
                
            pygame.draw.circle(screen, (255, 255, 0), (rx, ry), 5)

    # Drone
    if mode == 1:
        draw_arrow(screen, (255, 255, 255), CENTER, math.pi/2, size=12)
    else:
        dx, dy = world_to_radar(drone_pos[0], drone_pos[1])
        vec_x = dx - CENTER[0]
        vec_y = dy - CENTER[1]
        dist = math.sqrt(vec_x**2 + vec_y**2)
        max_r = (RADAR_SIZE // 2) - 8
        if dist > max_r:
            ratio = max_r / dist
            dx = int(CENTER[0] + vec_x * ratio)
            dy = int(CENTER[1] + vec_y * ratio)
        draw_arrow(screen, (255, 255, 255), (dx, dy), drone_yaw, size=12)

    font_radar = pygame.font.SysFont("monospace", 12, bold=True)
    mode_str = "HEAD UP" if mode == 1 else "NORTH UP"
    lbl = font_radar.render(f"{mode_str}", True, (200, 200, 200))
    screen.blit(lbl, (CENTER[0] - 30, CENTER[1] + RADAR_SIZE//2 + 5))

def save_data():
    if len(data_buffer["observations"]) == 0: return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    npz_filename = os.path.join(output_dir, f"flight_log_{timestamp}.npz")
    print(f"\nSaving {len(data_buffer['observations'])} steps to NPZ...")
    np.savez_compressed(
        npz_filename,
        obs=np.array(data_buffer["observations"]),
        actions=np.array(data_buffer["actions"]),
        rewards=np.array(data_buffer["rewards"]),
        dones=np.array(data_buffer["terminals"])
    )
    with open(os.path.join(output_dir, f"flight_log_{timestamp}.json"), "w") as f:
        json.dump(data_buffer, f, indent=4, cls=NumpyEncoder)
    print("Save Complete!")

# --- MAIN ---
pygame.init()
pygame.joystick.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Mission Recorder")
font = pygame.font.SysFont("monospace", 24, bold=True) # Larger Font
warning_font = pygame.font.SysFont("monospace", 60, bold=True) # Huge Warning
small_font = pygame.font.SysFont("monospace", 18)

if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

action = np.array([0.0, 0.0, 0.0, 0.0]) # Start Throttle 0
obs, _ = env.reset() 
draw_zone_boundary(ZONE_RADIUS)

current_targets = []
if isinstance(obs, dict) and "target_deltas" in obs:
    current_targets = obs["target_deltas"]

clock = pygame.time.Clock()
paused = True 
running = True

try:
    while running:
        clock.tick(TARGET_FPS)
        drone_id, pos, orn, euler = get_drone_state(env)
        pygame.event.pump() 
        
        # --- INPUTS ---
        raw_roll = joystick.get_axis(AXIS_ROLL)
        raw_pitch = joystick.get_axis(AXIS_PITCH)
        if INVERT_PITCH: raw_pitch = -raw_pitch
        raw_yaw = joystick.get_axis(AXIS_YAW)
        raw_throttle = joystick.get_axis(AXIS_THROTTLE)
        if INVERT_THROTTLE:
            thrust_cmd = (raw_throttle * -1.0 + 1.0) / 2.0
        else:
            thrust_cmd = (raw_throttle + 1.0) / 2.0

        def process_stick(val, expo, rate):
            if abs(val) < 0.05: val = 0.0
            val = apply_expo(val, expo)
            return val * rate

        action[0] = process_stick(raw_roll, EXPO_VALUE, MAX_ROLL_RATE)
        action[1] = process_stick(raw_pitch, EXPO_VALUE, MAX_PITCH_RATE)
        action[2] = process_stick(raw_yaw, EXPO_VALUE, MAX_YAW_RATE)
        action[3] = np.clip(thrust_cmd, 0.0, 1.0)

        # --- SIMULATION ---
        if not paused:
            data_buffer["observations"].append(obs)
            data_buffer["actions"].append(action.copy()) 

            obs, reward, terminated, truncated, info = env.step(action)
            
            if isinstance(obs, dict) and "target_deltas" in obs:
                current_targets = obs["target_deltas"]
            elif "target_deltas" in info:
                current_targets = info["target_deltas"]

            done = terminated or truncated
            data_buffer["rewards"].append(reward)
            data_buffer["terminals"].append(done)
            
            if len(data_buffer["observations"]) % LOG_FREQUENCY == 0:
                print(f"REC {len(data_buffer['observations']):04d} | [RADAR] Targets: {len(current_targets)}")

            if done: 
                print(">>> CRASH! Pausing...")
                obs, _ = env.reset()
                draw_zone_boundary(ZONE_RADIUS)
                action = np.array([0.0, 0.0, 0.0, 0.0])
                if isinstance(obs, dict) and "target_deltas" in obs:
                    current_targets = obs["target_deltas"]
                else:
                    current_targets = []
                paused = True 

        # --- RENDER ---
        main_surf = render_camera(drone_id, pos, orn, mode="chase", w=MAIN_RENDER_W, h=MAIN_RENDER_H)
        if main_surf:
            scaled_surf = pygame.transform.scale(main_surf, (WINDOW_W, WINDOW_H))
            screen.blit(scaled_surf, (0, 0))

        if not args.disable_hud:
            if SHOW_PIP and not paused:
                pip_surf = render_camera(drone_id, pos, orn, mode="fpv", w=PIP_SIZE[0], h=PIP_SIZE[1])
                if pip_surf:
                    pygame.draw.rect(screen, (255, 255, 255), (PIP_POS[0]-2, PIP_POS[1]-2, PIP_SIZE[0]+4, PIP_SIZE[1]+4))
                    screen.blit(pip_surf, PIP_POS)

            if euler and not args.no_horizon:
                roll_rad, pitch_rad = euler[0], euler[1]
                cx, cy = WINDOW_W // 2, WINDOW_H // 2
                pitch_offset = int(math.degrees(pitch_rad) * 6) 
                line_len = 400
                x1 = cx - line_len * math.cos(-roll_rad)
                y1 = (cy + pitch_offset) - line_len * math.sin(-roll_rad)
                x2 = cx + line_len * math.cos(-roll_rad)
                y2 = (cy + pitch_offset) + line_len * math.sin(-roll_rad)
                pygame.draw.line(screen, (0, 255, 255), (x1, y1), (x2, y2), 4)
                pygame.draw.circle(screen, (255, 0, 0), (cx, cy), 8) 

            if pos:
                dist = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                if dist > ZONE_RADIUS:
                    if (pygame.time.get_ticks() // 200) % 2 == 0: 
                        warn_surf = warning_font.render(f"OUT OF ZONE!", True, (255, 0, 0))
                        rect = warn_surf.get_rect(center=(WINDOW_W // 2, 100))
                        screen.blit(warn_surf, rect)
                
                draw_radar(screen, pos, euler[2], current_targets, ZONE_RADIUS, mode=RADAR_MODE)

                if args.show_data:
                    BOX_X, BOX_Y = 20, 80
                    s = pygame.Surface((300, 140))
                    s.set_alpha(150)
                    s.fill((0, 0, 0))
                    screen.blit(s, (BOX_X, BOX_Y))
                    pygame.draw.rect(screen, (255, 255, 255), (BOX_X, BOX_Y, 300, 140), 2)
                    
                    lines = [
                        f"ALT:   {pos[2]:.1f} m",
                        f"DIST:  {dist:.1f} / {ZONE_RADIUS:.0f}m",
                        f"ROLL:  {math.degrees(euler[0]):.1f}",
                        f"PITCH: {math.degrees(euler[1]):.1f}",
                        f"YAW:   {math.degrees(euler[2]):.1f}"
                    ]
                    for i, line in enumerate(lines):
                        color = (255, 50, 50) if (i == 1 and dist > ZONE_RADIUS * 0.9) else (0, 255, 0)
                        txt = small_font.render(line, True, color)
                        screen.blit(txt, (BOX_X + 20, BOX_Y + 15 + (i * 24)))

        throt_h = int(action[3] * 300)
        pygame.draw.rect(screen, (50, 50, 50), (20, WINDOW_H - 350, 40, 300)) 
        pygame.draw.rect(screen, (255, 0, 0), (20, WINDOW_H - 50 - throt_h, 40, throt_h))

        if paused:
            s = pygame.Surface((WINDOW_W, WINDOW_H))
            s.set_alpha(128)
            s.fill((0,0,0))
            screen.blit(s, (0,0))
            text = font.render("PAUSED - Btn 1 Resume", True, (255, 255, 0))
            screen.blit(text, (WINDOW_W//2 - 150, WINDOW_H//2))
        else:
            pygame.draw.circle(screen, (255, 0, 0), (WINDOW_W - 40, 40), 15)
            text = font.render("REC", True, (255, 0, 0))
            screen.blit(text, (WINDOW_W - 100, 30))
            count_text = font.render(f"Samples: {len(data_buffer['observations'])}", True, (0, 255, 0))
            screen.blit(count_text, (20, 20))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: paused = not paused
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == BTN_RESET: 
                    obs, _ = env.reset()
                    draw_zone_boundary(ZONE_RADIUS)
                    action = np.array([0.0, 0.0, 0.0, 0.0])
                    if isinstance(obs, dict) and "target_deltas" in obs:
                        current_targets = obs["target_deltas"]
                    else:
                        current_targets = []
                    paused = True
                if event.button == BTN_PAUSE: paused = not paused
                if event.button == BTN_PIP: SHOW_PIP = not SHOW_PIP
                if event.button == BTN_RADAR: RADAR_MODE = 1 - RADAR_MODE

except KeyboardInterrupt:
    print("Interrupted...")
finally:
    env.close()
    pygame.quit()
    save_data()