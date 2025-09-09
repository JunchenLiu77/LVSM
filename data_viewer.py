# basic viewer to visualize datasets
import json
import imageio.v3 as iio
import numpy as np
from pathlib import Path
from typing import List, Optional
import viser
import glob

class SceneViewer:
    def __init__(self, base_path: Path, scale_factor: float = 0.05, downsample_factor: int = 8):
        self.base_path = base_path
        self.scale_factor = scale_factor
        self.downsample_factor = downsample_factor
        self.server = viser.ViserServer()
        self.current_scene_idx = 0
        self.scene_paths: List[Path] = []
        self.frustums: List[viser.FrustumHandle] = []
        
        # Discover all scenes
        self._discover_scenes()
        
        # Add navigation controls
        self._add_navigation_controls()
        
        # Load first scene
        if self.scene_paths:
            self._load_scene(0)
    
    def _discover_scenes(self):
        """Discover all scene directories in the base path."""
        self.scene_paths = []
        for item in self.base_path.iterdir():
            if item.is_dir() and (item / "transforms.json").exists():
                self.scene_paths.append(item)
        
        self.scene_paths.sort()  # Sort for consistent ordering
        print(f"Found {len(self.scene_paths)} scenes in dataset")
    
    def _add_navigation_controls(self):
        """Add navigation buttons to the UI."""
        # Add folder for controls
        self.server.gui.add_folder("controls")
        
        # Previous scene button
        prev_button = self.server.gui.add_button("Previous Scene")
        @prev_button.on_click
        def _(_):
            if self.current_scene_idx > 0:
                self._load_scene(self.current_scene_idx - 1)
        
        # Next scene button
        next_button = self.server.gui.add_button("Next Scene")
        @next_button.on_click
        def _(_):
            if self.current_scene_idx < len(self.scene_paths) - 1:
                self._load_scene(self.current_scene_idx + 1)
        
        # Scene info display
        self.scene_info = self.server.gui.add_text("Scene Info", initial_value="")
    
    def _clear_frustums(self):
        """Clear all existing frustums from the scene."""
        for frustum in self.frustums:
            frustum.remove()
        self.frustums.clear()
    
    def _load_scene(self, scene_idx: int):
        """Load a specific scene by index."""
        if scene_idx < 0 or scene_idx >= len(self.scene_paths):
            return
        
        self.current_scene_idx = scene_idx
        scene_path = self.scene_paths[scene_idx]
        
        # Clear existing frustums
        self._clear_frustums()
        
        # Load the transforms.json file
        with open(scene_path / "transforms.json", "r") as f:
            transforms = json.load(f)
        
        # Extract camera parameters
        w = transforms["w"]
        h = transforms["h"]
        fl_x = transforms["fl_x"]
        fl_y = transforms["fl_y"]
        cx = transforms["cx"]
        cy = transforms["cy"]
        
        # Process each frame
        for i, frame_data in enumerate(transforms["frames"]):
            # Get transform matrix
            transform_matrix = np.array(frame_data["transform_matrix"])
            
            # Extract camera position and orientation
            camera_position = transform_matrix[:3, 3]
            camera_rotation = transform_matrix[:3, :3]
            
            # load the image
            image_path = scene_path / frame_data["file_path"]
            image_path = Path(str(image_path).replace("images/", "images_8/"))
            image = iio.imread(image_path)
            image = image[::self.downsample_factor, ::self.downsample_factor]
            
            aspect_ratio = w / h
            fov_x = 2 * np.arctan(w / (2 * fl_x))
            fov_y = 2 * np.arctan(h / (2 * fl_y))
            
            # Create a frame for this camera
            frustum = self.server.scene.add_camera_frustum(
                f"frustums/{i:05d}",
                fov = fov_x,
                aspect=aspect_ratio,
                scale=self.scale_factor,
                image=image,
                wxyz=viser.transforms.SO3.from_matrix(camera_rotation).wxyz,
                position=camera_position,
            )
            
            @frustum.on_click
            def _(_, frustum=frustum) -> None:
                for client in self.server.get_clients().values():
                    client.camera.wxyz = frustum.wxyz
                    client.camera.position = frustum.position
            
            self.frustums.append(frustum)
        
        # Update scene info
        scene_name = scene_path.name
        num_images = len(transforms["frames"])
        self.scene_info.value = f"Scene {scene_idx + 1}/{len(self.scene_paths)}: {scene_name}\nImages: {num_images}"
        
        print(f"Loaded scene {scene_idx + 1}/{len(self.scene_paths)}: {scene_name}")
        print(f"  - {num_images} images")
        print(f"  - Scene path: {scene_path}")
    
    def run(self):
        """Start the viewer and keep it running."""
        if not self.scene_paths:
            print("No scenes found in the dataset!")
            return
        
        print(f"Dataset contains {len(self.scene_paths)} scenes")
        print("Open your browser to http://localhost:8080 to view the visualization")
        print("Use the navigation buttons to switch between scenes")
        
        # Keep the server running
        input("Press Enter to exit...")

def main(
    base_path: Path = Path("/home/junchen/scratch/datasets/DL3DV-10K/1K/"),
    scale_factor: float = 0.05,
    downsample_factor: int = 8
) -> None:
    """
    Visualize multiple scenes in a dataset directory.
    
    Args:
        base_path (Path, optional): The base path containing multiple scene directories.
        scale_factor (float, optional): The scale factor for the frustums.
        downsample_factor (int, optional): The downsample factor for the images.
    """
    viewer = SceneViewer(base_path, scale_factor, downsample_factor)
    viewer.run()


if __name__ == "__main__":
    main()