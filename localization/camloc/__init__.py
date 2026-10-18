"""camloc: camera-based localization of the race car on the track map (iteration 1).

Modules
    allowed    which topics and car-state fields the method may read (challenge rules)
    bag        reading the mcap bag through that whitelist
    gt         ground truth from /tf, for map building, laps and scoring only
    geometry   SE(3) helpers
    trackmap   track borders: inside-track test, lateral offsets, ground height
    laps       start/finish crossings and the lap table
    odometry   causal car-state stream
    ekf        pose filter [x, y, yaw, gyro bias]
    reloc      relocalization against a feature map built from the training laps
    pipeline   the causal per-image-set loop
    metrics    the official score, per lap
"""
