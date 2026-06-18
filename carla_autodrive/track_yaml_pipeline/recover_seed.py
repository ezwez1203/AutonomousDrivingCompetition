import yaml

def recover_seed():
    with open("track_v7.yaml", "r") as f:
        data = yaml.safe_load(f)
        
    pts = data["centerline"]["points"]
    # Sample every 50 points (about 500mm spacing) to act as coarse seeds
    coarse_pts = pts[::50]
    
    seed_data = {
        "centerline": {
            "control_points": coarse_pts
        }
    }
    
    with open("track_seed.yaml", "w") as f:
        yaml.safe_dump(seed_data, f, sort_keys=False)

if __name__ == "__main__":
    recover_seed()
