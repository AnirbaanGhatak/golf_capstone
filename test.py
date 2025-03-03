from ultralytics.models.sam import SAM2VideoPredictor

# Create FastSAMPredictor
overrides = dict(conf=0.27, task="segment", mode="track", model="PTs/sam2.1_s.pt", save=False, imgsz=640)
predictor = SAM2VideoPredictor(overrides=overrides)

# Segment everything
everything_results = predictor("Test_videos/TV1_bh.mp4", points=[340, 340], labels=1)
