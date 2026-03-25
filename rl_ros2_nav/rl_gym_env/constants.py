# ── Constants ────────────────────────────────────────────────
LIDAR_SAMPLES   = 36        # downsample 360 rays → 36 (every 10°)
MAX_LIDAR_RANGE = 3.5       # metres — clip LiDAR beyond this
GOAL_X, GOAL_Y  = 1.5, 1    # goal position in the world (centre)
COLLISION_DIST  = 0.12      # metres — burger radius + small margin
MAX_STEPS       = 500       # episode time limit
SPAWN_AREA_SIZE = 3.0       # robot spawns within this square area around origin