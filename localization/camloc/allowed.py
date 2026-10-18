"""The one place that decides what the method may read from a bag (challenge rules, docs/DECISIONS.md D-001).

Inference code reads the bag only through camloc.bag, and camloc.bag reads only what is listed here.
Ground truth (/tf beyond the first pose, pose fields of state_estimation) is read by camloc.gt, which
is used for map building, lap definition and scoring, never inside the pipeline.
"""

CAMERAS = ('fl', 'fr', 'cl', 'cr', 'r', 'rl', 'rr')
IMAGE_TOPIC = '/constructor0/sensor/camera_{}/compressed_image'
INFO_TOPIC = '/constructor0/sensor/camera_{}/camera_info'
TF_STATIC_TOPIC = '/tf_static'
TRACK_TOPIC = '/constructor0/track_bounds'
STATE_TOPIC = '/constructor0/state_estimation'
FIRST_POSE_TOPIC = '/tf'          # only the first map -> base_link transform of the run

# Car-state fields that describe motion, not pose (INTRO slide 9: "speed, accelerations, yaw rate
# and others"). vx/vy are body-frame velocities: |vy| stays below a few m/s at 40 m/s (checked in
# scripts/prepare.py), so they carry no heading.
ALLOWED_STATE_FIELDS = (
    'vx_mps', 'vy_mps', 'v_mps', 'wz_radps', 'ax_mps2', 'ay_mps2',
)

# Pose and track-state fields: ground truth, never read at inference. yaw_vel_rad is listed on
# purpose: despite its name it equals yaw + beta (the absolute course angle), not a yaw rate.
FORBIDDEN_STATE_FIELDS = (
    'x_m', 'y_m', 'z_m', 'roll_rad', 'pitch_rad', 'yaw_rad', 'yaw_vel_rad', 'pos_accuracy',
    'sn_map_state',
)

assert not set(ALLOWED_STATE_FIELDS) & set(FORBIDDEN_STATE_FIELDS)
