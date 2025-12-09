import gymnasium as gym
import PyFlyt.gym_envs
import pygame
import numpy as np
import math
import datetime
import os
import json

# --- JOYSTICK MAPPING CONFIGURATION ---
AXIS_ROLL = 0
AXIS_PITCH = 1
AXIS_YAW = 2
AXIS_THROTTLE = 3

# --- BUTTON MAPPING ---
# Button 0 = Trigger
# Button 1 = Thumb Button (Side of stick)
# Button 2-5 = Top of stick
# Button 6-11 = Base of joystick
BTN_PAUSE = 1  # Side Thumb Button
BTN_PIP   = 2  # Top Stick Button
BTN_RESET = 7  # <--- UPDATED: Button 7 on Base

INVERT_PITCH = True
INVERT_THROTTLE = True

# --- SCREEN SETTINGS ---
MAIN_RENDER_W, MAIN_RENDER_H = 320, 240 
WINDOW_W, WINDOW_H = 800, 600
TARGET_FPS = 30 
LOG_FREQUENCY = 10 

# --- PIP SETTINGS ---
SHOW_PIP = True
PIP_SIZE = (240, 180)
PIP_POS = (WINDOW_W - 250, 10)

# --- DATA STORAGE ---
output_dir = "flight_data"
os.makedirs(output_dir, exist_ok=True)

data_buffer = {
    "observations": [],
    "actions": [],
    "rewards": [],
    "terminals": [] 
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

try:
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v4", render_mode="human")
except:
    env = gym.make("PyFlyt/Fixedwing-Waypoints-v0", render_mode="human")

try:
    if hasattr(env.unwrapped, 'ctx'):
        p = env.unwrapped.ctx.pybullet_client
    elif hasattr(env.unwrapped, 'env'):
        p = env.unwrapped.env.aviary.ctx.pybullet_client
    else:
        import pybullet as p
except:
    import pybullet as p

def get_drone_id(env):
    try:
        return env.unwrapped.env.drones[0].Id
    except:
        return None

def render_camera(drone_id, mode="chase", w=320, h=240):
    if drone_id is None: return None
    try:
        pos, orn = p.getBasePositionAndOrientation(drone_id)
    except:
        return None 

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

def save_data():
    if len(data_buffer["observations"]) == 0:
        print("No data to save.")
        return

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

    json_filename = os.path.join(output_dir, f"flight_log_{timestamp}.json")
    print(f"Saving JSON...")
    with open(json_filename, "w") as f:
        json.dump(data_buffer, f, indent=4, cls=NumpyEncoder)
        
    print(f"SUCCESS: Data saved to {output_dir}")

# --- SETUP ---
pygame.init()
pygame.joystick.init()
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("Recorder - Btn 1: Pause | Btn 7 (Base): Reset")
font = pygame.font.SysFont("monospace", 20, bold=True)

if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
else:
    print("CRITICAL: No Joystick Found!")
    exit()

action = np.array([0.0, 0.0, 0.0, 0.5])
obs, _ = env.reset() 
clock = pygame.time.Clock()

paused = True 
running = True

try:
    while running:
        clock.tick(TARGET_FPS)
        drone_id = get_drone_id(env)
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

        def deadzone(val): return val if abs(val) > 0.05 else 0.0

        action[0] = deadzone(raw_roll)
        action[1] = deadzone(raw_pitch)
        action[2] = deadzone(raw_yaw)
        action[3] = np.clip(thrust_cmd, 0.0, 1.0)

        # --- SIMULATION STEP ---
        if not paused:
            data_buffer["observations"].append(obs)
            data_buffer["actions"].append(action.copy()) 

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            data_buffer["rewards"].append(reward)
            data_buffer["terminals"].append(done)
            
            sample_count = len(data_buffer["observations"])
            if sample_count % LOG_FREQUENCY == 0:
                act_str = np.array2string(action, precision=2, suppress_small=True, separator=',')
                print(f"REC #{sample_count:04d} | Act: {act_str} | Rew: {reward:.4f}")

            if done: 
                print(">>> CRASH/RESET. Pausing Simulation.")
                obs, _ = env.reset()
                action = np.array([0.0, 0.0, 0.0, 0.5])
                paused = True 

        # --- RENDER ---
        main_surf = render_camera(drone_id, mode="chase", w=MAIN_RENDER_W, h=MAIN_RENDER_H)
        if main_surf:
            scaled_surf = pygame.transform.scale(main_surf, (WINDOW_W, WINDOW_H))
            screen.blit(scaled_surf, (0, 0))

        if SHOW_PIP and not paused:
            pip_surf = render_camera(drone_id, mode="fpv", w=PIP_SIZE[0], h=PIP_SIZE[1])
            if pip_surf:
                pygame.draw.rect(screen, (255, 255, 255), (PIP_POS[0]-2, PIP_POS[1]-2, PIP_SIZE[0]+4, PIP_SIZE[1]+4))
                screen.blit(pip_surf, PIP_POS)

        # --- HUD ---
        cx, cy = WINDOW_W // 2, WINDOW_H // 2
        pygame.draw.line(screen, (0, 255, 0), (cx-20, cy), (cx+20, cy), 2)
        pygame.draw.line(screen, (0, 255, 0), (cx, cy-20), (cx, cy+20), 2)
        
        throt_h = int(action[3] * 200)
        pygame.draw.rect(screen, (50, 50, 50), (20, WINDOW_H - 250, 30, 200)) 
        pygame.draw.rect(screen, (255, 0, 0), (20, WINDOW_H - 50 - throt_h, 30, throt_h))

        if paused:
            s = pygame.Surface((WINDOW_W, WINDOW_H))
            s.set_alpha(128)
            s.fill((0,0,0))
            screen.blit(s, (0,0))
            text = font.render("PAUSED - Press Button 1 to Resume", True, (255, 255, 0))
            screen.blit(text, (WINDOW_W//2 - 200, WINDOW_H//2))
        else:
            pygame.draw.circle(screen, (255, 0, 0), (WINDOW_W - 30, 30), 10)
            text = font.render("REC", True, (255, 0, 0))
            screen.blit(text, (WINDOW_W - 70, 20))
            count_text = font.render(f"Samples: {len(data_buffer['observations'])}", True, (0, 255, 0))
            screen.blit(count_text, (20, 20))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == BTN_RESET: 
                    obs, _ = env.reset()
                    action[3] = 0.5
                    paused = True
                
                if event.button == BTN_PAUSE: 
                    paused = not paused
                
                if event.button == BTN_PIP: 
                    SHOW_PIP = not SHOW_PIP

except KeyboardInterrupt:
    print("Interrupted...")
finally:
    env.close()
    pygame.quit()
    save_data()
