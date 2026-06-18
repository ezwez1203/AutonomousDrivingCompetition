import yaml
import cv2
import numpy as np
from track_yaml_pipeline import PipelineConfig, segment_target_line, auto_detect_frame, red_candidate_points

def fix_seed():
    cfg = PipelineConfig(
        image_path="../../circuit_blueprint.png",
        output_yaml="dummy.yaml",
        output_overlay="dummy.png",
        seed_yaml="track_seed.yaml",
        field_width_mm=16000.0,
        field_height_mm=12000.0,
        road_width_mm=850.0,
        reference="inner"
    )
    bgr = cv2.imread(cfg.image_path, cv2.IMREAD_COLOR)
    mask = segment_target_line(bgr, cfg)
    frame = auto_detect_frame(bgr, cfg, mask)
    candidates = red_candidate_points(mask)
    
    with open("track_seed.yaml", "r") as f:
        data = yaml.safe_load(f)
    
    points = data["centerline"]["control_points"]
    
    # Create a density map (count of red pixels in a 21x21 neighborhood)
    mask_float = (mask > 0).astype(np.float32)
    density_map = cv2.boxFilter(mask_float, -1, (21, 21), normalize=False)
    
    for pt in points:
        p_mm = (pt["x"], pt["y"])
        p_px = np.asarray(frame.mm_to_px(p_mm), dtype=np.float64)
        
        d2 = np.sum((candidates - p_px) ** 2, axis=1)
        valid_idx = np.where(d2 < 80**2)[0] # Search within 80px (about 250mm)
        
        if len(valid_idx) == 0:
            nearest = candidates[np.argmin(d2)]
        else:
            local_cands = candidates[valid_idx]
            cx = np.clip(np.round(local_cands[:, 0]).astype(int), 0, density_map.shape[1]-1)
            cy = np.clip(np.round(local_cands[:, 1]).astype(int), 0, density_map.shape[0]-1)
            densities = density_map[cy, cx]
            
            # Within 80px, pick the pixel with the absolute highest density 
            # (i.e. the thickest part of the line)
            best_local_idx = np.argmax(densities)
            nearest = local_cands[best_local_idx]
        
        new_mm = frame.px_to_mm((nearest[0], nearest[1]))
        pt["x"] = float(round(new_mm[0], 1))
        pt["y"] = float(round(new_mm[1], 1))
        
    with open("track_seed.yaml", "w") as f:
        # Save as the same file to directly update the seed
        yaml.safe_dump(data, f, sort_keys=False)

if __name__ == "__main__":
    fix_seed()
