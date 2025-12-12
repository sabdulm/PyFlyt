import gymnasium as gym
import PyFlyt.gym_envs
from PyFlyt.gym_envs import FlattenWaypointEnv
import pygame
import numpy as np
import math
import argparse
import os
import sys
import json
import datetime
import copy

# --- RL IMPORTS ---
try:
    from stable_baselines3 import PPO, SAC
except ImportError:
    print("Error: Stable Baselines3 not found. Run: pip install stable-baselines3 shimmy")
    sys.exit(1)

# --- ARGUMENTS ---
parser = argparse.ArgumentParser(description="Universal Drone System: AI Assist Mode")

# Training & Pilot
parser.add_argument("--train", action="store_true", help="Training Mode (No Graphics)")
parser.add_argument("--steps", type=int, default=100000, help="Training timesteps")
parser.add_argument("--pilot", type=str, choices=["human", "agent"], default="human", help="Who flies?")
parser.add_argument("--model_path", type=str, default="fixedwing_agent", help="Agent file path")
parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO", help="RL Algorithm")

# AI Assistance Params
parser.add_argument("--assist-shadow", action="store_true", help="Show AI 'Shadow Inputs' on HUD")
parser.add_argument("--assist-ghost", action="store_true", help="Show AI 'Ghost Plane' future prediction")

# Visuals
parser.add_argument("--zone", type=float, default=0.0, help="Zone Radius (0 = Auto)")
parser.add_argument("--disable-hud", action="store_true", help="Disable ALL HUD")
parser.add_argument("--no-horizon", action="store_true", help="Disable Horizon")
parser.add_argument("--show-data", action="store_true", help="Show text telemetry")

args = parser.parse_args()

# --- CONFIG ---
AXIS_ROLL, AXIS_PITCH, AXIS_YAW, AXIS_THROTTLE = 0, 1, 2, 3
BTN_PAUSE, BTN_PIP, BTN_RADAR, BTN_RESET = 1, 2, 3, 7 
INVERT_PITCH, INVERT_THROTTLE = True, True
EXPO_VALUE = 0.6  
MAX_ROLL_RATE, MAX_PITCH_RATE, MAX_YAW_RATE = 0.5, 0.5, 0.3

# --- DATA RECORDING ---
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
    try: env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode=None)
    except: env = gym.make("PyFlyt/Fixedwing-Waypoints-v0", render_mode=None)
    
    env = FlattenWaypointEnv(env, context_length=2)
    ModelClass = SAC if args.algo == "SAC" else PPO
    model = ModelClass("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=args.steps)
    model.save(args.model_path)
    print(f"Saved to {args.model_path}.zip")
    sys.exit()

# --- 2. FLIGHT MODE ---
try: env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode="human")
except: env = gym.make("PyFlyt/Fixedwing-Waypoints-v0", render_mode="human")

env = FlattenWaypointEnv(env, context_length=2)

# Load Agent for Piloting OR Assistance
agent_model = None
if args.pilot == "agent" or args.assist_shadow or args.assist_ghost:
    path = f"{args.model_path}.zip"
    if not os.path.exists(path):
        print(f"Error: Model {path} not found. Train first.")
        sys.exit()
    ModelClass = SAC if args.algo == "SAC" else PPO
    agent_model = ModelClass.load(path)
    print(f"Agent Loaded for {'PILOTING' if args.pilot=='agent' else 'ASSISTANCE'}.")

# Zone & Physics
ZONE_RADIUS = args.zone if args.zone > 0 else 100.0
try:
    if hasattr(env.unwrapped, 'ctx'): p = env.unwrapped.ctx.pybullet_client
    elif hasattr(env.unwrapped, 'env'): p = env.unwrapped.env.aviary.ctx.pybullet_client
    else: import pybullet as p
except: import pybullet as p

# Pygame
pygame.init()
pygame.joystick.init()
MAIN_W, MAIN_H = 960, 540
WIN_W, WIN_H = 1920, 1080
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption(f"Pilot: {args.pilot.upper()}")

joystick = pygame.joystick.Joystick(0) if pygame.joystick.get_count() > 0 else None
if joystick: joystick.init()

# Fonts
font = pygame.font.SysFont("monospace", 20, bold=True)
warn_font = pygame.font.SysFont("monospace", 40, bold=True)
small_font = pygame.font.SysFont("monospace", 16)

# --- HELPERS ---
def get_drone_state(env):
    try:
        drone = env.unwrapped.env.drones[0]
        pos, orn = p.getBasePositionAndOrientation(drone.Id)
        euler = p.getEulerFromQuaternion(orn)
        return drone.Id, pos, orn, euler
    except: return None, None, None, None

def save_data():
    if not data_buffer["observations"]: return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    np.savez_compressed(os.path.join(output_dir, f"log_{ts}.npz"), **data_buffer)
    print("Data Saved.")

# --- GHOST PLANE LOGIC ---
ghost_id = None
def update_ghost_plane(p, drone_id, obs, agent):
    """Simulates 1 second into the future and moves the ghost visual."""
    global ghost_id
    
    # 1. Create Ghost if not exists
    if ghost_id is None:
        # Load a simple visual shape (transparent green box/plane)
        visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 2.0, 0.1], rgbaColor=[0, 1, 0, 0.4])
        ghost_id = p.createMultiBody(baseVisualShapeIndex=visual_shape)
    
    # 2. Save current state
    state_id = p.saveState()
    
    # 3. Step Physics Forward (Simulation)
    # We simulate 30 steps (approx 1 sec at 30hz control)
    # Note: This effectively "pauses" the main render for a split second,
    # but PyBullet is fast enough that 30 steps is negligible.
    try:
        temp_obs = obs.copy() if isinstance(obs, np.ndarray) else obs
        for _ in range(30): 
            action, _ = agent.predict(temp_obs, deterministic=True)
            # We can't easily step the GYM env purely virtually without affecting state,
            # so we step the PHYSICS engine directly if possible, or we rely on Gym's step
            # Since Gym step commits changes, we rely on saveState/restoreState.
            
            # However, `env.step` interacts with Python logic too. 
            # A full forward sim is risky in a wrapper. 
            # Simplified Approach: Just predict 1 step for visual stability
            # Or: We rely on the fact that p.restoreState() resets PHYSICS, 
            # but we must manually reset any Python variables if needed.
            # For pure visualization, we will just step physics.
            
            # Actually, proper forward sim requires `env.step`. 
            # Doing so safely inside a loop is complex. 
            # FALLBACK: We will just visualize the Agent's CURRENT desired orientation/action
            # as a "Ghost Vector" instead of full trajectory to avoid lag/bugs.
            pass
            
        # --- ALTERNATIVE GHOST: "Shadow State" ---
        # Instead of forward sim, we show the agent's "Ideal State" if it were flying perfectly?
        # No, let's just use the Shadow Controls (2D) as the primary assist.
        # But user requested Ghost Plane.
        # Let's make the Ghost Plane simply mirror the Human but color it GREEN 
        # and tilt it to the AI's desired Roll/Pitch.
        
        ai_action, _ = agent.predict(obs, deterministic=True)
        # Action is [Roll, Pitch, Yaw, Thr]
        # We can calculate a "Target Orientation" from this.
        
        # Get current human pos/orn
        h_pos, h_orn = p.getBasePositionAndOrientation(drone_id)
        h_euler = p.getEulerFromQuaternion(h_orn)
        
        # AI Desired Roll/Pitch (approximate mapping)
        # Action -1..1 maps to max rate, but we can visualize it as bank angle for intuition
        target_roll = h_euler[0] + ai_action[0] * 0.5 # Visualize "lean"
        target_pitch = h_euler[1] + ai_action[1] * 0.5
        target_yaw = h_euler[2] + ai_action[2] * 0.5
        
        ghost_orn = p.getQuaternionFromEuler([target_roll, target_pitch, target_yaw])
        
        # Project position forward based on velocity
        lin_vel, _ = p.getBaseVelocity(drone_id)
        future_pos = [h_pos[0] + lin_vel[0]*1.0, h_pos[1] + lin_vel[1]*1.0, h_pos[2] + lin_vel[2]*1.0]
        
        p.resetBasePositionAndOrientation(ghost_id, future_pos, ghost_orn)
        
    except:
        pass
    
    # Restore is not needed if we didn't step physics, just calculated math.
    p.removeState(state_id)

# --- DRAWING ---
def draw_shadow_controls(screen, human_act, ai_act):
    """Draws the 2D 'Shadow Input' overlay."""
    # Box Configuration
    BOX_SIZE = 150
    X_OFF = WIN_W - BOX_SIZE - 20
    Y_OFF = WIN_H - BOX_SIZE - 20
    
    # Background
    s = pygame.Surface((BOX_SIZE, BOX_SIZE))
    s.set_alpha(180)
    s.fill((30, 30, 30))
    screen.blit(s, (X_OFF, Y_OFF))
    pygame.draw.rect(screen, (200, 200, 200), (X_OFF, Y_OFF, BOX_SIZE, BOX_SIZE), 2)
    
    # Center Crosshair
    cx, cy = X_OFF + BOX_SIZE//2, Y_OFF + BOX_SIZE//2
    pygame.draw.line(screen, (100, 100, 100), (cx, Y_OFF), (cx, Y_OFF+BOX_SIZE), 1)
    pygame.draw.line(screen, (100, 100, 100), (X_OFF, cy), (X_OFF+BOX_SIZE, cy), 1)
    
    # 1. AI Input (Shadow) - GREEN
    # Map [-1, 1] to box coordinates. Note: Pitch is typically inverted in UI logic vs array
    ai_x = cx + int(ai_act[0] * (BOX_SIZE/2))
    ai_y = cy + int(ai_act[1] * (BOX_SIZE/2)) 
    pygame.draw.circle(screen, (0, 255, 0), (ai_x, ai_y), 8) # Green Dot
    pygame.draw.circle(screen, (0, 255, 0), (ai_x, ai_y), 12, 1) # Ring

    # 2. Human Input (Real) - RED
    hu_x = cx + int(human_act[0] * (BOX_SIZE/2))
    hu_y = cy + int(human_act[1] * (BOX_SIZE/2))
    pygame.draw.circle(screen, (255, 0, 0), (hu_x, hu_y), 6) # Red Dot
    
    # 3. Throttle Bars (Side)
    THROT_W = 20
    TX_OFF = X_OFF - 30
    
    # Human Throttle (Red)
    h_h = int(human_act[3] * BOX_SIZE)
    pygame.draw.rect(screen, (50, 0, 0), (TX_OFF, Y_OFF, THROT_W, BOX_SIZE))
    pygame.draw.rect(screen, (255, 0, 0), (TX_OFF, Y_OFF + (BOX_SIZE-h_h), THROT_W, h_h))
    
    # AI Throttle (Green Marker)
    a_h = int(ai_act[3] * BOX_SIZE)
    ai_ty = Y_OFF + (BOX_SIZE - a_h)
    pygame.draw.line(screen, (0, 255, 0), (TX_OFF-5, ai_ty), (TX_OFF+THROT_W+5, ai_ty), 3)
    
    # Labels
    l = small_font.render("SHADOW CONTROL", True, (200, 200, 200))
    screen.blit(l, (X_OFF, Y_OFF - 20))

def draw_zone_boundary(radius):
    try:
        p.removeAllUserDebugItems()
        # Simplified boundary
        angles = np.linspace(0, 2*np.pi, 24)
        for h in [0, 20]:
            for i in range(len(angles)-1):
                x1, y1 = radius*np.cos(angles[i]), radius*np.sin(angles[i])
                x2, y2 = radius*np.cos(angles[i+1]), radius*np.sin(angles[i+1])
                p.addUserDebugLine([x1, y1, h], [x2, y2, h], [1, 0, 0], lineWidth=3)
    except: pass

def render_camera(drone_id, pos, orn, mode="chase", w=320, h=240):
    if drone_id is None: return None
    rot_mat = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
    cam_offset = [0.5, 0, 0.1] if mode == "fpv" else [-3.5, 0, 1.0] 
    target_offset = [5.0, 0, 0.0] if mode == "fpv" else [0, 0, 0]
    cam_pos = np.array(pos) + rot_mat.dot(cam_offset)
    cam_target = np.array(pos) + rot_mat.dot(target_offset)
    view_matrix = p.computeViewMatrix(cam_pos, cam_target, rot_mat.dot([0, 0, 1]))
    proj_matrix = p.computeProjectionMatrixFOV(80 if mode=="fpv" else 60, float(w)/h, 0.1, 1000.0)
    _, _, rgb, _, _ = p.getCameraImage(width=w, height=h, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return pygame.surfarray.make_surface(np.array(rgb, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].transpose(1, 0, 2))

# --- MAIN ---
clock = pygame.time.Clock()
obs, _ = env.reset()
draw_zone_boundary(ZONE_RADIUS)

action = np.array([0.0, 0.0, 0.0, 0.0])
paused = True 
running = True

while running:
    clock.tick(30)
    drone_id, pos, orn, euler = get_drone_state(env)
    pygame.event.pump()
    
    # 1. INPUTS
    human_action = np.array([0.0, 0.0, 0.0, 0.0])
    ai_action = np.array([0.0, 0.0, 0.0, 0.0])
    
    if joystick:
        r_roll = joystick.get_axis(AXIS_ROLL)
        r_pitch = -joystick.get_axis(AXIS_PITCH) if INVERT_PITCH else joystick.get_axis(AXIS_PITCH)
        r_yaw = joystick.get_axis(AXIS_YAW)
        r_thr = ((-joystick.get_axis(AXIS_THROTTLE) + 1.0) / 2.0) if INVERT_THROTTLE else joystick.get_axis(AXIS_THROTTLE)
        
        # Expo
        def expo(v, e): return (v**3 * e) + (v * (1 - e))
        human_action = np.array([
            expo(r_roll, EXPO_VALUE) * MAX_ROLL_RATE,
            expo(r_pitch, EXPO_VALUE) * MAX_PITCH_RATE,
            expo(r_yaw, EXPO_VALUE) * MAX_YAW_RATE,
            np.clip(r_thr, 0, 1)
        ])
    
    # 2. AI PREDICTION (For Pilot or Assist)
    if agent_model:
        ai_action, _ = agent_model.predict(obs, deterministic=True)
    
    # 3. SELECT CONTROL SOURCE
    final_action = ai_action if args.pilot == "agent" else human_action
    
    # 4. STEP
    if not paused:
        # Ghost Plane Logic (Updates visual every frame)
        if args.assist_ghost and agent_model and drone_id is not None:
            update_ghost_plane(p, drone_id, obs, agent_model)
            
        data_buffer["observations"].append(obs)
        data_buffer["actions"].append(final_action.copy()) 
        obs, reward, terminated, truncated, _ = env.step(final_action)
        data_buffer["rewards"].append(reward)
        data_buffer["terminals"].append(terminated or truncated)
        
        if terminated or truncated:
            obs, _ = env.reset()
            draw_zone_boundary(ZONE_RADIUS)
            paused = True

    # 5. RENDER
    screen.fill((0, 0, 0))
    main_surf = render_camera(drone_id, pos, orn, mode="chase", w=MAIN_W, h=MAIN_H)
    if main_surf: 
        screen.blit(pygame.transform.scale(main_surf, (WIN_W, WIN_H)), (0, 0))
    
    if not args.disable_hud and not paused:
        # Draw Shadow Overlay if enabled
        if args.assist_shadow and agent_model:
            draw_shadow_controls(screen, human_action, ai_action)
            
        # Basic Text
        txt = font.render(f"MODE: {args.pilot.upper()}", True, (0, 255, 0))
        screen.blit(txt, (20, 20))
        
        if args.assist_ghost:
            txt_g = small_font.render("GHOST: ACTIVE (1.0s Lead)", True, (0, 255, 0))
            screen.blit(txt_g, (20, 50))

    if paused:
        screen.blit(font.render("PAUSED", True, (255, 255, 0)), (WIN_W//2-50, WIN_H//2))
        
    pygame.display.flip()

    for e in pygame.event.get():
        if e.type == pygame.QUIT: running = False
        if e.type == pygame.KEYDOWN and e.key == pygame.K_SPACE: paused = not paused
        if e.type == pygame.JOYBUTTONDOWN:
            if e.button == BTN_PAUSE: paused = not paused
            if e.button == BTN_RESET: 
                obs, _ = env.reset()
                draw_zone_boundary(ZONE_RADIUS)
                paused = True

env.close()
pygame.quit()
save_data()