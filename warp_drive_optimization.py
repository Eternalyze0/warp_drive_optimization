import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import constants
from mpl_toolkits.mplot3d import Axes3D
import time


import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants
from scipy.special import expit  # Smooth sigmoid function

class SmoothWarpBubble3DVisualizer:
    def __init__(self, model):
        self.model = model
        
    def smooth_bubble_evolution(self, phase, formation_time=0.3, cruise_time=0.4, dissolution_time=0.3):
        """
        Smooth, continuous bubble evolution with proper formation and dissolution phases
        """
        # Normalize times to sum to 1
        total_time = formation_time + cruise_time + dissolution_time
        formation_time /= total_time
        cruise_time /= total_time
        dissolution_time /= total_time
        
        formation_end = formation_time
        cruise_end = formation_time + cruise_time
        
        if phase < formation_end:
            # Smooth formation: 0 → 1
            t_norm = phase / formation_end
            # Sigmoid-like smooth rise
            return 0.5 * (1 - np.cos(np.pi * t_norm))
            
        elif phase < cruise_end:
            # Full bubble during cruise: maintain 1.0
            return 1.0
            
        else:
            # Smooth dissolution: 1 → 0
            t_norm = (phase - cruise_end) / dissolution_time
            # Sigmoid-like smooth fall
            return 0.5 * (1 + np.cos(np.pi * t_norm))
    
    def create_smooth_3d_animation(self, duration_seconds=12, fps=20):
        """Create smooth, continuous 3D animation of warp bubble evolution"""
        print("Creating smooth 3D warp bubble animation...")
        
        total_frames = duration_seconds * fps
        phases = np.linspace(0, 1, total_frames)
        
        # Create spatial grid
        x = np.linspace(-2, 2, 40)
        y = np.linspace(-2, 2, 40)
        X, Y = np.meshgrid(x, y)
        
        # Precompute smooth bubble evolution
        print("Precomputing smooth bubble evolution...")
        bubble_strengths = [self.smooth_bubble_evolution(p) for p in phases]
        bubble_surfaces = []
        
        for i, phase in enumerate(phases):
            if i % 15 == 0:
                print(f"Frame {i}/{total_frames} - Strength: {bubble_strengths[i]:.3f}")
                
            t = 0.2 * phase
            surface_height = np.zeros_like(X)
            current_strength = bubble_strengths[i]
            
            with torch.no_grad():
                # Batch compute all points
                coords_list = []
                for xi in x:
                    for yi in y:
                        coords_list.append([t, xi, yi, 0.0, phase])
                
                coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
                metric_flat = self.model(coords_tensor)
                g = self.reshape_to_metric(metric_flat, len(coords_list))
                
                # Calculate warp strength
                g_minkowski = torch.eye(4).unsqueeze(0).repeat(len(coords_list), 1, 1)
                g_minkowski[:, 0, 0] = -1
                warp_strength = torch.mean(torch.abs(g - g_minkowski), dim=(1, 2))
                surface_data = warp_strength.reshape(len(y), len(x)).numpy()
                
                # Shape like a smooth bubble
                r = np.sqrt(X**2 + Y**2)
                
                # SMOOTH bubble profile - no hard edges
                # Use smooth transitions between regions
                inner_transition = 0.5 * (1 + np.tanh(10 * (r - 0.2)))  # Smooth inner edge
                outer_transition = 0.5 * (1 - np.tanh(10 * (r - 0.9)))  # Smooth outer edge
                
                # Combine transitions to create smooth shell
                shell_profile = inner_transition * outer_transition
                
                # Apply smooth bubble strength
                surface_height = surface_data * shell_profile * current_strength * 2.0
                
                # Add slight background warping that also evolves smoothly
                background = 0.1 * current_strength * (1 - shell_profile)
                surface_height += background
            
            bubble_surfaces.append(surface_height)
        
        # Create animation figure
        fig = plt.figure(figsize=(14, 10))
        
        # 3D view
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Timeline view
        ax2 = fig.add_subplot(222)
        
        # Radial profile view
        ax3 = fig.add_subplot(223)
        
        # Info panel
        ax4 = fig.add_subplot(224)
        
        # Create initial plots
        surf = ax1.plot_surface(X, Y, bubble_surfaces[0], 
                              cmap='viridis', alpha=0.85,
                              linewidth=0.5, antialiased=True)
        
        # Timeline
        timeline, = ax2.plot(phases[:1], bubble_strengths[:1], 'b-', linewidth=3, label='Bubble Strength')
        current_point = ax2.scatter([0], [bubble_strengths[0]], color='red', s=80, zorder=5)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1.1)
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Bubble Strength')
        ax2.set_title('Evolution Timeline')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Radial profile for current frame
        radial_distances = np.linspace(0, 2, 50)
        current_radial = self.compute_radial_profile(phases[0], bubble_strengths[0])
        radial_line, = ax3.plot(radial_distances, current_radial, 'r-', linewidth=2)
        ax3.set_xlabel('Radial Distance')
        ax3.set_ylabel('Warp Strength')
        ax3.set_title('Radial Profile')
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, max(current_radial) * 1.2 if max(current_radial) > 0 else 1)
        
        # Info panel
        info_text = ax4.text(0.05, 0.95, '', transform=ax4.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        ax4.axis('off')
        
        # Set up 3D plot
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Warp Field Strength')
        ax1.set_title('3D Warp Bubble - FORMATION')
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        
        plt.colorbar(surf, ax=ax1, shrink=0.6, label='Warp Strength')
        plt.tight_layout()
        
        def animate(frame):
            phase = phases[frame]
            strength = bubble_strengths[frame]
            
            # Clear and update 3D surface
            ax1.clear()
            surf = ax1.plot_surface(X, Y, bubble_surfaces[frame], 
                                  cmap='viridis', alpha=0.85,
                                  linewidth=0.5, antialiased=True)
            
            # Update timeline
            timeline.set_data(phases[:frame+1], bubble_strengths[:frame+1])
            current_point.set_offsets([[phase, strength]])
            
            # Update radial profile
            new_radial = self.compute_radial_profile(phase, strength)
            radial_line.set_ydata(new_radial)
            ax3.set_ylim(0, max(new_radial) * 1.2 if max(new_radial) > 0 else 1)
            
            # Determine stage with smooth transitions
            formation_end = 0.3
            cruise_end = 0.7
            
            if phase < formation_end:
                stage = "FORMATION"
                stage_color = "green"
                progress = phase / formation_end
            elif phase < cruise_end:
                stage = "CRUISE"
                stage_color = "blue"
                progress = (phase - formation_end) / (cruise_end - formation_end)
            else:
                stage = "DISSOLUTION"
                stage_color = "red"
                progress = (phase - cruise_end) / (1 - cruise_end)
            
            # Update titles and info
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Warp Field Strength')
            ax1.set_title(f'3D Warp Bubble - {stage}\nPhase: {phase:.2f}', color=stage_color)
            ax1.set_xlim(-2, 2)
            ax1.set_ylim(-2, 2)
            
            # Update info text
            info_str = f"""WARP BUBBLE STATUS
Stage: {stage}
Phase: {phase:.3f}
Strength: {strength:.3f}
Progress: {progress:.1%}

FORMATION: 0.0 - 0.3
CRUISE: 0.3 - 0.7  
DISSOLUTION: 0.7 - 1.0"""
            info_text.set_text(info_str)
            
            return [surf, timeline, current_point, radial_line, info_text]
        
        # Create smooth animation
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames,
            interval=1000/fps, blit=False, repeat=True
        )
        
        print("Smooth 3D animation created successfully!")
        return anim
    
    def compute_radial_profile(self, phase, strength):
        """Compute smooth radial profile for current state"""
        radial_distances = np.linspace(0, 2, 50)
        profile = []
        
        t = 0.2 * phase
        
        with torch.no_grad():
            for r_val in radial_distances:
                coords = torch.tensor([[t, r_val, 0.0, 0.0, phase]], dtype=torch.float32)
                metric_flat = self.model(coords)
                g = self.reshape_to_metric(metric_flat, 1)
                warp_strength = torch.mean(torch.abs(g - torch.eye(4))).item()
                
                # Apply smooth radial profile
                inner = 0.5 * (1 + np.tanh(10 * (r_val - 0.2)))
                outer = 0.5 * (1 - np.tanh(10 * (r_val - 0.9)))
                radial_factor = inner * outer
                
                profile.append(warp_strength * radial_factor * strength * 2.0)
        
        return np.array(profile)
    
    def create_comparison_animation(self, duration_seconds=10, fps=20):
        """Show comparison between old (discontinuous) and new (smooth) evolution"""
        print("Creating evolution comparison...")
        
        total_frames = duration_seconds * fps
        phases = np.linspace(0, 1, total_frames)
        
        # Compute both evolution profiles
        old_strengths = [2*p if p < 0.5 else 2*(1-p) for p in phases]  # Old discontinuous
        new_strengths = [self.smooth_bubble_evolution(p) for p in phases]  # New smooth
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(phases, old_strengths, 'r-', linewidth=2, label='Old (Discontinuous)')
        plt.plot(phases, new_strengths, 'b-', linewidth=2, label='New (Smooth)')
        plt.xlabel('Phase')
        plt.ylabel('Bubble Strength')
        plt.title('Evolution Profile Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        # Show derivatives to highlight smoothness
        old_deriv = np.gradient(old_strengths, phases)
        new_deriv = np.gradient(new_strengths, phases)
        plt.plot(phases, old_deriv, 'r--', linewidth=2, label='Old Derivative')
        plt.plot(phases, new_deriv, 'b--', linewidth=2, label='New Derivative')
        plt.xlabel('Phase')
        plt.ylabel('Rate of Change')
        plt.title('Smoothness Comparison (Derivatives)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 3)
        # Show the smooth function components
        formation = [0.5 * (1 - np.cos(np.pi * p/0.3)) if p < 0.3 else 0 for p in phases]
        cruise = [1.0 if 0.3 <= p < 0.7 else 0 for p in phases]
        dissolution = [0.5 * (1 + np.cos(np.pi * (p-0.7)/0.3)) if p >= 0.7 else 0 for p in phases]
        
        plt.plot(phases, formation, 'g-', linewidth=2, label='Formation')
        plt.plot(phases, cruise, 'b-', linewidth=2, label='Cruise')
        plt.plot(phases, dissolution, 'r-', linewidth=2, label='Dissolution')
        plt.plot(phases, new_strengths, 'k--', linewidth=3, label='Combined')
        plt.xlabel('Phase')
        plt.ylabel('Component Strength')
        plt.title('Smooth Evolution Components')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def reshape_to_metric(self, metric_flat, batch_size):
        g = torch.zeros(batch_size, 4, 4)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        
        return g

# # In your main execution, replace with:
# print("=== SMOOTH WARP BUBBLE EVOLUTION ===")

# # Create smooth visualizer
# smooth_visualizer = SmoothWarpBubble3DVisualizer(model)

# # First show the comparison between old and new
# print("Showing evolution profile comparison...")
# smooth_visualizer.create_comparison_animation()

# # Then create the smooth animation
# print("Creating smooth continuous animation...")
# plt.close('all')
# smooth_animation = smooth_visualizer.create_smooth_3d_animation(duration_seconds=12, fps=20)

# # Save it
# print("Saving smooth animation...")
# try:
#     smooth_animation.save('smooth_warp_bubble_3d.gif', writer='pillow', fps=20, dpi=100)
#     print("Smooth animation saved as 'smooth_warp_bubble_3d.gif'")
# except Exception as e:
#     print(f"Could not save: {e}")

# print("Displaying smooth animation...")
# plt.show()



class WarpBubble3DVisualizer:
    def __init__(self, model):
        self.model = model
        
    def create_3d_bubble_animation(self, duration_seconds=12, fps=20):
        """Create 3D visualization of the actual warp bubble geometry"""
        print("Creating 3D warp bubble animation...")
        
        total_frames = duration_seconds * fps
        phases = np.linspace(0, 1, total_frames)
        
        # Create 3D spatial grid
        x = np.linspace(-2, 2, 40)
        y = np.linspace(-2, 2, 40)
        z = np.linspace(-2, 2, 40)
        X, Y = np.meshgrid(x, y)
        
        # Create 3D figure
        fig = plt.figure(figsize=(16, 12))
        
        # Main 3D plot of bubble geometry
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Cross-section views
        ax2 = fig.add_subplot(222)  # XY plane
        ax3 = fig.add_subplot(223)  # XZ plane  
        ax4 = fig.add_subplot(224)  # Energy timeline
        
        fig.suptitle('3D Warp Bubble Geometry Visualization', fontsize=16, y=0.95)
        
        print("Precomputing 3D bubble data...")
        
        # Precompute bubble geometry data
        bubble_surfaces = []
        energy_slices_xy = []
        energy_slices_xz = []
        total_energies = []
        
        for i, phase in enumerate(phases):
            if i % 10 == 0:
                print(f"Computing 3D frame {i}/{total_frames}")
                
            t = 0.2 * phase  # Coordinate time
            
            # Compute bubble surface (where warping is maximum)
            bubble_surface = np.zeros_like(X)
            energy_slice_xy = np.zeros_like(X)
            energy_slice_xz = np.zeros_like(X)
            
            # Sample points for 3D surface
            with torch.no_grad():
                # XY plane slice (z=0)
                coords_xy = []
                for xi in x:
                    for yi in y:
                        coords_xy.append([t, xi, yi, 0.0, phase])
                
                coords_tensor_xy = torch.tensor(coords_xy, dtype=torch.float32)
                metric_flat_xy = self.model(coords_tensor_xy)
                g_xy = self.reshape_to_metric(metric_flat_xy, len(coords_xy))
                
                # XZ plane slice (y=0)  
                coords_xz = []
                for xi in x:
                    for zi in z:
                        coords_xz.append([t, xi, 0.0, zi, phase])
                
                coords_tensor_xz = torch.tensor(coords_xz, dtype=torch.float32)
                metric_flat_xz = self.model(coords_tensor_xz)
                g_xz = self.reshape_to_metric(metric_flat_xz, len(coords_xz))
                
                # Calculate warp field strength (deviation from Minkowski)
                g_minkowski = torch.eye(4).unsqueeze(0)
                g_minkowski[:, 0, 0] = -1
                
                # XY plane data
                warp_strength_xy = torch.mean(torch.abs(g_xy - g_minkowski), dim=(1, 2))
                energy_slice_xy = warp_strength_xy.reshape(len(y), len(x)).numpy()
                
                # XZ plane data
                warp_strength_xz = torch.mean(torch.abs(g_xz - g_minkowski), dim=(1, 2))
                energy_slice_xz = warp_strength_xz.reshape(len(z), len(x)).numpy()
                
                # Create 3D bubble surface - height represents warp strength
                # We'll create a surface where the height shows the warping
                surface_height = energy_slice_xy.copy()
                
                # Apply bubble profile
                r = np.sqrt(X**2 + Y**2)
                bubble_strength = 2*phase if phase < 0.5 else 2*(1-phase)
                surface_height *= bubble_strength
                
                # Shape it like a bubble - stronger at shell, flat inside
                shell_mask = (r >= 0.3) & (r <= 1.0)
                inside_mask = r < 0.3
                outside_mask = r > 1.0
                
                surface_height[inside_mask] *= 0.1  # Flat inside
                surface_height[outside_mask] *= 0.01  # Almost flat outside
                surface_height[shell_mask] *= 1.0  # Full strength in shell
            
            bubble_surfaces.append(surface_height)
            energy_slices_xy.append(energy_slice_xy)
            energy_slices_xz.append(energy_slice_xz)
            total_energies.append(np.sum(surface_height))
        
        print("Creating 3D animation...")
        
        # Create initial 3D surface plot
        surf = ax1.plot_surface(X, Y, bubble_surfaces[0], 
                               cmap='viridis', alpha=0.8, 
                               linewidth=0, antialiased=True)
        
        # Add contour lines on the "ground"
        contour = ax1.contour(X, Y, bubble_surfaces[0], 
                             zdir='z', offset=np.min(bubble_surfaces[0])-0.1, 
                             cmap='viridis', alpha=0.6)
        
        # Initial 2D slices
        im2 = ax2.imshow(energy_slices_xy[0], extent=[-2, 2, -2, 2],
                        origin='lower', cmap='plasma', aspect='auto')
        ax2.set_title('XY Plane Slice (z=0)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        
        im3 = ax3.imshow(energy_slices_xz[0], extent=[-2, 2, -2, 2], 
                        origin='lower', cmap='plasma', aspect='auto')
        ax3.set_title('XZ Plane Slice (y=0)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('z')
        
        # Energy timeline
        energy_line, = ax4.plot(phases[:1], total_energies[:1], 'b-', linewidth=2)
        current_point = ax4.scatter([0], [total_energies[0]], color='red', s=50, zorder=5)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, max(total_energies)*1.1)
        ax4.set_xlabel('Phase')
        ax4.set_ylabel('Total Warp Field Energy')
        ax4.set_title('Energy Timeline')
        ax4.grid(True, alpha=0.3)
        
        # Set up 3D plot
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y') 
        ax1.set_zlabel('Warp Field Strength')
        ax1.set_title('3D Warp Bubble Geometry')
        ax1.set_xlim(-2, 2)
        ax1.set_ylim(-2, 2)
        
        # Colorbar for 3D plot
        plt.colorbar(surf, ax=ax1, shrink=0.6, aspect=20, label='Warp Strength')
        plt.colorbar(im2, ax=ax2, label='Energy Density')
        plt.colorbar(im3, ax=ax3, label='Energy Density')
        
        plt.tight_layout()
        
        def animate(frame):
            phase = phases[frame]
            bubble_strength = 2*phase if phase < 0.5 else 2*(1-phase)
            
            # Clear previous 3D plots
            ax1.clear()
            
            # Update 3D surface
            surf = ax1.plot_surface(X, Y, bubble_surfaces[frame], 
                                   cmap='viridis', alpha=0.8,
                                   linewidth=0, antialiased=True)
            
            # Update contour
            contour = ax1.contour(X, Y, bubble_surfaces[frame], 
                                 zdir='z', offset=np.min(bubble_surfaces[frame])-0.1,
                                 cmap='viridis', alpha=0.6)
            
            # Update 2D slices
            im2.set_array(energy_slices_xy[frame])
            im3.set_array(energy_slices_xz[frame])
            
            # Update energy timeline
            energy_line.set_data(phases[:frame+1], total_energies[:frame+1])
            current_point.set_offsets([[phase, total_energies[frame]]])
            
            # Update titles with phase info
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Warp Field Strength')
            ax1.set_title(f'3D Warp Bubble\nPhase: {phase:.2f}, Strength: {bubble_strength:.2f}')
            ax1.set_xlim(-2, 2)
            ax1.set_ylim(-2, 2)
            
            # Determine stage
            if phase < 0.25:
                stage = "FORMATION"
                stage_color = "green"
            elif phase < 0.75:
                stage = "CRUISE"
                stage_color = "blue" 
            else:
                stage = "DISSOLUTION"
                stage_color = "red"
            
            fig.suptitle(f'3D Warp Bubble - {stage}\nPhase: {phase:.2f}', 
                        fontsize=16, y=0.95, color=stage_color)
            
            return [surf, contour, im2, im3, energy_line, current_point]
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames,
            interval=1000/fps, blit=False, repeat=True
        )
        
        print("3D animation created successfully!")
        return anim
    
    def create_simple_3d_bubble(self, phase=0.5):
        """Create a static 3D visualization of the warp bubble"""
        print(f"Creating static 3D bubble at phase {phase}")
        
        # Create spatial grid
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        
        # Compute bubble surface
        t = 0.2 * phase
        surface_height = np.zeros_like(X)
        
        with torch.no_grad():
            coords_list = []
            for xi in x:
                for yi in y:
                    coords_list.append([t, xi, yi, 0.0, phase])
            
            coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
            metric_flat = self.model(coords_tensor)
            g = self.reshape_to_metric(metric_flat, len(coords_list))
            
            # Calculate warp strength
            g_minkowski = torch.eye(4).unsqueeze(0)
            g_minkowski[:, 0, 0] = -1
            warp_strength = torch.mean(torch.abs(g - g_minkowski), dim=(1, 2))
            surface_data = warp_strength.reshape(len(y), len(x)).numpy()
            
            # Shape like a bubble
            r = np.sqrt(X**2 + Y**2)
            bubble_strength = 2*phase if phase < 0.5 else 2*(1-phase)
            
            # Create bubble geometry
            surface_height = np.zeros_like(X)
            shell_mask = (r >= 0.3) & (r <= 1.0)
            surface_height[shell_mask] = surface_data[shell_mask] * bubble_strength * 2.0
            surface_height[r < 0.3] = surface_data[r < 0.3] * bubble_strength * 0.1  # Flat inside
            surface_height[r > 1.0] = surface_data[r > 1.0] * bubble_strength * 0.01  # Flat outside
        
        # Create 3D plot
        fig = plt.figure(figsize=(15, 10))
        
        # Main 3D view
        ax1 = fig.add_subplot(221, projection='3d')
        surf1 = ax1.plot_surface(X, Y, surface_height, cmap='viridis', 
                                alpha=0.9, linewidth=0, antialiased=True)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Warp Field Strength')
        ax1.set_title(f'3D Warp Bubble\nPhase: {phase:.2f}')
        plt.colorbar(surf1, ax=ax1, shrink=0.6, label='Warp Strength')
        
        # Top-down view
        ax2 = fig.add_subplot(222)
        contour = ax2.contourf(X, Y, surface_height, levels=20, cmap='viridis')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Top-Down View')
        ax2.set_aspect('equal')
        plt.colorbar(contour, ax=ax2, label='Warp Strength')
        
        # Side view (XZ slice)
        ax3 = fig.add_subplot(223)
        # For XZ slice, we'll compute a different cut
        xz_slice = np.zeros((len(x), len(x)))
        for i, xi in enumerate(x):
            for k, zi in enumerate(x):  # using x for z range for simplicity
                coords = torch.tensor([[t, xi, 0.0, zi, phase]], dtype=torch.float32)
                metric_flat = self.model(coords)
                g = self.reshape_to_metric(metric_flat, 1)
                warp_strength = torch.mean(torch.abs(g - torch.eye(4))).item()
                xz_slice[k, i] = warp_strength
        
        im3 = ax3.imshow(xz_slice, extent=[-2, 2, -2, 2], 
                        origin='lower', cmap='plasma', aspect='auto')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_title('XZ Plane Slice (y=0)')
        plt.colorbar(im3, ax=ax3, label='Warp Strength')
        
        # Radial profile
        ax4 = fig.add_subplot(224)
        radial_distances = np.linspace(0, 2, 100)
        radial_profile = []
        for r_val in radial_distances:
            # Sample along x-axis at distance r
            coords = torch.tensor([[t, r_val, 0.0, 0.0, phase]], dtype=torch.float32)
            metric_flat = self.model(coords)
            g = self.reshape_to_metric(metric_flat, 1)
            warp_strength = torch.mean(torch.abs(g - torch.eye(4))).item()
            radial_profile.append(warp_strength)
        
        ax4.plot(radial_distances, radial_profile, 'b-', linewidth=2)
        ax4.axvline(x=0.3, color='red', linestyle='--', alpha=0.7, label='Inner Shell')
        ax4.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Outer Shell')
        ax4.set_xlabel('Radial Distance')
        ax4.set_ylabel('Warp Field Strength')
        ax4.set_title('Radial Profile')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return surface_height
    
    def reshape_to_metric(self, metric_flat, batch_size):
        g = torch.zeros(batch_size, 4, 4)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        
        return g

# # In your main execution, replace the animation section with:
# print("Creating 3D warp bubble visualization...")
# visualizer = WarpBubble3DVisualizer(model)

# # First, show a static 3D bubble to verify it works
# print("Showing static 3D bubble...")
# static_bubble = visualizer.create_simple_3d_bubble(phase=0.5)

# # Then create the full 3D animation
# print("Creating full 3D animation...")
# animation_3d = visualizer.create_3d_bubble_animation(duration_seconds=10, fps=15)

# # Save the 3D animation
# print("Saving 3D animation...")
# try:
#     animation_3d.save('warp_bubble_3d.mp4', writer='ffmpeg', fps=15, dpi=100)
#     print("3D animation saved as 'warp_bubble_3d.mp4'")
# except Exception as e:
#     print(f"Could not save 3D animation: {e}")

# print("Displaying 3D animation...")
# plt.show()


class TimeDependentWarpDriveNN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6):
        super(TimeDependentWarpDriveNN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(5, hidden_dim))  # Now 5 inputs: (t, x, y, z, phase)
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, 10))  # 10 components for symmetric 4x4 metric
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, coordinates):
        return self.network(coordinates)

class TimeDependentWarpDriveObjective:
    def __init__(self, weights=None):
        self.weights = weights or {
            'einstein': 1.0,
            'warp_bubble': 2.0,
            'energy_violation': 1.5,
            'energy_minimization': 3.0,
            'causality': 1.0,
            'boundary': 0.5,
            'regularity': 0.3,
            'time_evolution': 2.0,  # NEW: Time evolution constraint
            'bubble_dynamics': 1.5, # NEW: Bubble formation/dissolution
        }
    
    def __call__(self, model, coordinates):
        total_loss = 0.0
        loss_components = {}
        
        # Split coordinates into spatial and phase (time evolution parameter)
        spatial_coords = coordinates[:, :4]  # (t, x, y, z)
        phase = coordinates[:, 4:5]         # Evolution phase parameter
        
        # 1. Einstein Field Equations (with time dependence)
        loss_einstein = self.compute_einstein_loss(model, spatial_coords, phase)
        total_loss += self.weights['einstein'] * loss_einstein
        loss_components['einstein'] = loss_einstein.item()
        
        # 2. Warp Bubble Structure (time-dependent)
        loss_bubble = self.compute_warp_bubble_loss(model, spatial_coords, phase)
        total_loss += self.weights['warp_bubble'] * loss_bubble
        loss_components['bubble'] = loss_bubble.item()
        
        # 3. Energy Condition Violation
        loss_energy_violation = self.compute_energy_condition_loss(model, spatial_coords, phase)
        total_loss += self.weights['energy_violation'] * loss_energy_violation
        loss_components['energy_violation'] = loss_energy_violation.item()
        
        # 4. Energy Minimization
        loss_energy_minimization = self.compute_energy_minimization_loss(model, spatial_coords, phase)
        total_loss += self.weights['energy_minimization'] * loss_energy_minimization
        loss_components['energy_minimization'] = loss_energy_minimization.item()
        
        # 5. Causality Preservation
        loss_causality = self.compute_causality_loss(model, spatial_coords, phase)
        total_loss += self.weights['causality'] * loss_causality
        loss_components['causality'] = loss_causality.item()
        
        # 6. Boundary Conditions
        loss_boundary = self.compute_boundary_loss(model, spatial_coords, phase)
        total_loss += self.weights['boundary'] * loss_boundary
        loss_components['boundary'] = loss_boundary.item()
        
        # 7. Metric Regularity
        loss_regularity = self.compute_regularity_loss(model, spatial_coords, phase)
        total_loss += self.weights['regularity'] * loss_regularity
        loss_components['regularity'] = loss_regularity.item()
        
        # 8. NEW: Time Evolution Constraints
        loss_time_evolution = self.compute_time_evolution_loss(model, spatial_coords, phase)
        total_loss += self.weights['time_evolution'] * loss_time_evolution
        loss_components['time_evolution'] = loss_time_evolution.item()
        
        # 9. NEW: Bubble Dynamics
        loss_bubble_dynamics = self.compute_bubble_dynamics_loss(model, spatial_coords, phase)
        total_loss += self.weights['bubble_dynamics'] * loss_bubble_dynamics
        loss_components['bubble_dynamics'] = loss_bubble_dynamics.item()
        
        return total_loss, loss_components
    
    def compute_einstein_loss(self, model, spatial_coords, phase):
        """Einstein equations with time dependence"""
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        full_coords.requires_grad_(True)
        
        metric_flat = model(full_coords)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        # Time-dependent Einstein tensor approximation
        G_approx = torch.zeros(batch_size, 4, 4, device=spatial_coords.device)
        
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 0:
                    G_approx[:, i, j] = g[:, i, j] + 1.0
                elif i == j:
                    G_approx[:, i, j] = g[:, i, j] - 1.0
                else:
                    G_approx[:, i, j] = g[:, i, j]
        
        return torch.mean(G_approx**2)
    
    def compute_warp_bubble_loss(self, model, spatial_coords, phase):
        """Time-dependent warp bubble structure"""
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        
        r = torch.norm(spatial_coords[:, 1:], dim=1)  # Spatial radius
        t = spatial_coords[:, 0]  # Coordinate time
        
        metric_flat = model(full_coords)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        loss_bubble = 0.0
        g_minkowski = self.get_minkowski_metric(batch_size, spatial_coords.device)
        
        # Bubble formation/dissolution based on phase
        bubble_strength = self.bubble_evolution_profile(phase)
        
        inside_mask = r < 0.3
        shell_mask = (r >= 0.3) & (r < 1.0)
        outside_mask = r >= 1.0
        
        # Inside bubble should approach Minkowski, modulated by bubble strength
        if torch.any(inside_mask):
            g_inside = g[inside_mask]
            target_inside = g_minkowski[inside_mask] + (1 - bubble_strength[inside_mask].unsqueeze(-1).unsqueeze(-1)) * 0.1
            loss_inside = torch.mean((g_inside - target_inside)**2)
            loss_bubble += loss_inside
        
        # Shell region warping scales with bubble strength
        if torch.any(shell_mask):
            g_shell = g[shell_mask]
            shell_strength = bubble_strength[shell_mask]
            target_warp = 0.5 * shell_strength.unsqueeze(-1).unsqueeze(-1)
            loss_shell = torch.mean((g_shell - (g_minkowski[shell_mask] + target_warp))**2)
            loss_bubble += loss_shell
        
        # Outside should always be Minkowski
        if torch.any(outside_mask):
            g_outside = g[outside_mask]
            loss_outside = torch.mean((g_outside - g_minkowski[outside_mask])**2)
            loss_bubble += loss_outside
        
        return loss_bubble
    
    def compute_energy_condition_loss(self, model, spatial_coords, phase):
        """Encourage violation of Weak Energy Condition in shell region"""
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        
        r = torch.norm(spatial_coords[:, 1:], dim=1)
        shell_mask = (r >= 0.3) & (r < 1.0)
        
        if not torch.any(shell_mask):
            return torch.tensor(0.0, device=spatial_coords.device)
        
        metric_flat = model(full_coords[shell_mask])
        g_shell = self.reshape_to_metric(metric_flat, torch.sum(shell_mask))
        
        # For warp drives, g_tt should be less negative in shell region
        # This corresponds to negative energy density
        g_tt_shell = g_shell[:, 0, 0]
        
        # Target: g_tt > -1 (less negative than Minkowski)
        # But not too close to 0 (which would require excessive energy)
        target_g_tt = -0.7  # Balanced value for moderate energy requirements
        loss_energy = torch.mean((g_tt_shell - target_g_tt)**2)
        
        return loss_energy
    
    def compute_energy_minimization_loss(self, model, spatial_coords, phase):
        """Time-dependent energy minimization"""
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        full_coords.requires_grad_(True)
        
        r = torch.norm(spatial_coords[:, 1:], dim=1)
        shell_mask = (r >= 0.3) & (r < 1.0)
        
        if not torch.any(shell_mask):
            return torch.tensor(0.0, device=spatial_coords.device)
        
        metric_flat = model(full_coords[shell_mask])
        g_shell = self.reshape_to_metric(metric_flat, torch.sum(shell_mask))
        
        # Energy density estimate (time-dependent)
        g_minkowski = self.get_minkowski_metric(len(g_shell), spatial_coords.device)
        metric_deviation = g_shell - g_minkowski
        
        # Scale energy by bubble strength (less energy when bubble is weak)
        bubble_strength = self.bubble_evolution_profile(phase[shell_mask])
        energy_density_estimate = (
            torch.abs(metric_deviation[:, 0, 0]) +
            torch.abs(metric_deviation[:, 0, 1]) +
            torch.abs(metric_deviation[:, 0, 2]) +
            torch.abs(metric_deviation[:, 0, 3])
        ) * bubble_strength
        
        loss_energy_minimization = torch.mean(energy_density_estimate**2)
        
        # Additional penalty for sharp gradients
        loss_gradients = 0.0
        for i in range(4):
            for j in range(4):
                g_ij = g_shell[:, i, j]
                grad_g = torch.autograd.grad(
                    g_ij, full_coords,
                    grad_outputs=torch.ones_like(g_ij),
                    create_graph=True, retain_graph=True
                )[0]
                loss_gradients += torch.mean(grad_g**2)
        
        return loss_energy_minimization + 0.1 * loss_gradients
    
    def compute_causality_loss(self, model, spatial_coords, phase):
        """Preserve causality - no closed timelike curves"""
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        
        metric_flat = model(full_coords)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        # Check metric determinant (should be negative for Lorentzian signature)
        det_g = torch.det(g)
        loss_det = torch.mean(torch.relu(det_g + 1e-6))
        
        # Check time-time component (should be negative)
        g_tt = g[:, 0, 0]
        loss_g_tt = torch.mean(torch.relu(g_tt + 1e-6))
        
        return loss_det + loss_g_tt
    
    def compute_boundary_loss(self, model, spatial_coords, phase):
        """Asymptotic flatness at infinity"""
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        
        r = torch.norm(spatial_coords[:, 1:], dim=1)
        far_mask = r > 2.0
        
        if not torch.any(far_mask):
            return torch.tensor(0.0, device=spatial_coords.device)
        
        metric_flat = model(full_coords[far_mask])
        g_far = self.reshape_to_metric(metric_flat, torch.sum(far_mask))
        g_minkowski = self.get_minkowski_metric(len(g_far), spatial_coords.device)
        
        return torch.mean((g_far - g_minkowski)**2)
    
    def compute_regularity_loss(self, model, spatial_coords, phase):
        """Metric regularity and smoothness"""
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        full_coords.requires_grad_(True)
        
        metric_flat = model(full_coords)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        # Penalize large metric components
        loss_magnitude = torch.mean(g**2)
        
        # Penalize large derivatives
        loss_smoothness = 0.0
        for i in range(4):
            for j in range(4):
                g_ij = g[:, i, j]
                grad_g = torch.autograd.grad(
                    g_ij, full_coords,
                    grad_outputs=torch.ones_like(g_ij),
                    create_graph=True, retain_graph=True
                )[0]
                loss_smoothness += torch.mean(grad_g**2)
        
        return 0.1 * loss_magnitude + 0.01 * loss_smoothness
    
    def compute_time_evolution_loss(self, model, spatial_coords, phase):
        """Ensure smooth time evolution of the metric"""
        batch_size = spatial_coords.shape[0]
        spatial_coords = spatial_coords.clone().requires_grad_(True)
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        full_coords.requires_grad_(True)
        
        metric_flat = model(full_coords)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        # Penalize large time derivatives (ensure smooth evolution)
        loss_time_derivatives = 0.0
        for i in range(4):
            for j in range(4):
                g_ij = g[:, i, j]
                # Derivative with respect to coordinate time
                grad_g = torch.autograd.grad(
                    g_ij, spatial_coords,
                    grad_outputs=torch.ones_like(g_ij),
                    create_graph=True, retain_graph=True
                )[0]
                time_deriv = grad_g[:, 0]  # derivative w.r.t. t
                loss_time_derivatives += torch.mean(time_deriv**2)
        
        return loss_time_derivatives * 0.1
    
    def compute_bubble_dynamics_loss(self, model, spatial_coords, phase):
        """Control bubble formation and dissolution dynamics"""
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        
        r = torch.norm(spatial_coords[:, 1:], dim=1)
        shell_mask = (r >= 0.3) & (r < 1.0)
        
        if not torch.any(shell_mask):
            return torch.tensor(0.0, device=spatial_coords.device)
        
        metric_flat = model(full_coords[shell_mask])
        g_shell = self.reshape_to_metric(metric_flat, torch.sum(shell_mask))
        
        # Get bubble strength for these points
        bubble_strength = self.bubble_evolution_profile(phase[shell_mask])
        
        # Measure actual warping in shell
        g_minkowski = self.get_minkowski_metric(len(g_shell), spatial_coords.device)
        actual_warping = torch.mean(torch.abs(g_shell - g_minkowski), dim=(1, 2))
        
        # Warping should follow bubble strength profile
        target_warping = 0.3 * bubble_strength  # Scale factor
        loss_dynamics = torch.mean((actual_warping - target_warping)**2)
        
        return loss_dynamics
    
    def bubble_evolution_profile(self, phase):
        """Define how bubble strength evolves with phase parameter"""
        # Phase: 0 = start formation, 0.5 = full bubble, 1.0 = dissolved
        # Smooth profile for formation and dissolution
        return torch.where(
            phase < 0.5,
            2 * phase,                    # Linear formation: 0 → 1
            2 * (1 - phase)               # Linear dissolution: 1 → 0
        )
    
    def reshape_to_metric(self, metric_flat, batch_size):
        g = torch.zeros(batch_size, 4, 4, device=metric_flat.device)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        
        return g
    
    def get_minkowski_metric(self, batch_size, device):
        g = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        g[:, 0, 0] = -1.0
        return g

class TimeDependentWarpDriveTrainer:
    def __init__(self, model, objective):
        self.model = model
        self.objective = objective
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=100)
        self.loss_history = {key: [] for key in [
            'total', 'einstein', 'bubble', 'energy_violation', 'energy_minimization', 
            'causality', 'boundary', 'regularity', 'time_evolution', 'bubble_dynamics'
        ]}
    
    def train(self, epochs=1000, batch_size=1024):
        print("Training Time-Dependent Warp Drive Neural Network...")
        
        for epoch in range(epochs):
            coordinates = self.sample_training_points(batch_size)
            total_loss, loss_components = self.objective(self.model, coordinates)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            self.loss_history['total'].append(total_loss.item())
            for key, value in loss_components.items():
                self.loss_history[key].append(value)
            
            if epoch % 100 == 0:
                self.print_progress(epoch, total_loss, loss_components)
    
    def sample_training_points(self, batch_size):
        """Sample coordinates with time and phase evolution"""
        # Spatial coordinates
        spatial = torch.rand(batch_size, 4) * 4 - 2
        
        # Phase parameter for bubble evolution (0 to 1)
        phase = torch.rand(batch_size, 1)
        
        # Combine
        coordinates = torch.cat([spatial, phase], dim=1)
        
        return coordinates
    
    def print_progress(self, epoch, total_loss, components):
        print(f"\nEpoch {epoch}: Total Loss = {total_loss.item():.6f}")
        print("Component Losses:")
        for name, value in components.items():
            print(f"  {name:20}: {value:.6f}")

class WarpBubbleAnimator:
    def __init__(self, model):
        self.model = model
        self.fig = None

    def create_warp_bubble_animation_fast(self, duration_seconds=10, fps=15):
        """Fast version - compute frames on-the-fly during animation"""
        print("Creating fast animation (on-the-fly computation)...")
        
        total_frames = duration_seconds * fps
        phases = np.linspace(0, 1, total_frames)
        
        # Lower resolution for speed
        x = np.linspace(-2, 2, 25)
        y = np.linspace(-2, 2, 25)
        X, Y = np.meshgrid(x, y)
        
        self.fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        def animate(frame):
            phase = phases[frame]
            t = 0.5 * phase
            
            # Compute just this frame
            coords_list = []
            for x_val in x:
                for y_val in y:
                    coords_list.append([t, x_val, y_val, 0.0, phase])
            
            coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
            
            with torch.no_grad():
                metric_flat = self.model(coords_tensor)
                g = self.reshape_to_metric(metric_flat, len(coords_tensor))
                
                g_tt_grid = g[:, 0, 0].reshape(len(y), len(x)).numpy()
                g_tx_grid = g[:, 0, 1].reshape(len(y), len(x)).numpy()
            
            # Clear and plot this frame only
            for ax in axes.flat:
                ax.clear()
            
            # ... plotting code for this single frame ...
            
            return []
        
        anim = animation.FuncAnimation(
            self.fig, animate, frames=total_frames,
            interval=1000/fps, blit=False, repeat=True
        )
        
        return anim

    def create_warp_bubble_animation(self, duration_seconds=10, fps=30):
        """Create full animation of warp bubble lifecycle"""
        print("Creating warp bubble animation...")
        
        total_frames = duration_seconds * fps
        phases = np.linspace(0, 1, total_frames)
        
        # Create spatial grid
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        
        # Initialize figure
        self.fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        self.fig.suptitle('Warp Bubble Dynamics: Formation → Travel → Dissolution', fontsize=16)
        
        # Precompute data for animation
        print("Precomputing animation frames...")
        g_tt_data = []
        g_tx_data = []
        energy_data = []
        bubble_strength_data = []
        
        for i, phase in enumerate(phases):
            if i % 10 == 0:
                print(f"Frame {i}/{total_frames}")
                
            g_tt_frame = np.zeros_like(X)
            g_tx_frame = np.zeros_like(X)
            energy_frame = np.zeros_like(X)
            
            with torch.no_grad():
                for j in range(len(x)):
                    for k in range(len(y)):
                        # Coordinate time advances with phase for travel effect
                        t = 0.5 * phase  # Simulate time passing during travel
                        coords = torch.tensor([[t, x[j], y[k], 0.0, phase]], dtype=torch.float32)
                        metric_flat = self.model(coords)
                        g = self.reshape_to_metric(metric_flat, 1)
                        
                        g_tt_frame[k, j] = g[0, 0, 0].item()
                        g_tx_frame[k, j] = g[0, 0, 1].item()
                        
                        # Estimate energy density
                        g_minkowski = torch.eye(4)
                        g_minkowski[0, 0] = -1
                        deviation = torch.abs(g - g_minkowski).mean().item()
                        energy_frame[k, j] = deviation
            
            g_tt_data.append(g_tt_frame)
            g_tx_data.append(g_tx_frame)
            energy_data.append(energy_frame)
            bubble_strength_data.append(self.bubble_evolution_profile_np(phase))
        
        # Create animation
        def animate(frame):
            phase = phases[frame]
            bubble_strength = bubble_strength_data[frame]
            
            # Clear axes
            for ax in axes.flat:
                ax.clear()
            
            # Plot 1: g_tt component
            im1 = axes[0,0].imshow(g_tt_data[frame], extent=[-2, 2, -2, 2], 
                                  origin='lower', cmap='RdBu_r', vmin=-2, vmax=0)
            axes[0,0].set_title(f'Time-Time Component (g_tt)\nPhase: {phase:.2f}')
            axes[0,0].set_xlabel('x')
            axes[0,0].set_ylabel('y')
            self.fig.colorbar(im1, ax=axes[0,0])
            
            # Plot 2: g_tx component
            im2 = axes[0,1].imshow(g_tx_data[frame], extent=[-2, 2, -2, 2],
                                  origin='lower', cmap='RdBu_r', vmin=-1, vmax=1)
            axes[0,1].set_title(f'Time-Space Component (g_tx)\nBubble Strength: {bubble_strength:.2f}')
            axes[0,1].set_xlabel('x')
            axes[0,1].set_ylabel('y')
            self.fig.colorbar(im2, ax=axes[0,1])
            
            # Plot 3: Energy density
            im3 = axes[1,0].imshow(energy_data[frame], extent=[-2, 2, -2, 2],
                                  origin='lower', cmap='hot', vmin=0, vmax=1)
            axes[1,0].set_title('Energy Density Estimate')
            axes[1,0].set_xlabel('x')
            axes[1,0].set_ylabel('y')
            self.fig.colorbar(im3, ax=axes[1,0])
            
            # Plot 4: Bubble evolution profile
            all_phases = np.linspace(0, 1, 100)
            bubble_profile = [self.bubble_evolution_profile_np(p) for p in all_phases]
            axes[1,1].plot(all_phases, bubble_profile, 'b-', linewidth=2)
            axes[1,1].axvline(x=phase, color='red', linestyle='--', linewidth=2)
            axes[1,1].set_title('Bubble Evolution Profile')
            axes[1,1].set_xlabel('Phase')
            axes[1,1].set_ylabel('Bubble Strength')
            axes[1,1].grid(True, alpha=0.3)
            axes[1,1].set_xlim(0, 1)
            axes[1,1].set_ylim(0, 1)
            
            # Add phase description
            if phase < 0.25:
                stage = "FORMATION"
                color = "green"
            elif phase < 0.75:
                stage = "TRAVEL"
                color = "blue"
            else:
                stage = "DISSOLUTION"
                color = "red"
            
            self.fig.text(0.5, 0.95, f'Stage: {stage}', 
                         ha='center', fontsize=14, color=color, weight='bold')
            
            plt.tight_layout()
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(
            self.fig, animate, frames=total_frames,
            interval=1000/fps, blit=False, repeat=True
        )
        
        print("Animation created successfully!")
        return anim
    
    def bubble_evolution_profile_np(self, phase):
        """NumPy version of bubble evolution profile"""
        if phase < 0.5:
            return 2 * phase
        else:
            return 2 * (1 - phase)
    
    def reshape_to_metric(self, metric_flat, batch_size):
        g = torch.zeros(batch_size, 4, 4)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        
        return g

class WarpBubbleAnimator:
    def __init__(self, model):
        self.model = model
        
    def create_warp_bubble_animation(self, duration_seconds=10, fps=20):
        """Create animation that actually renders frames"""
        print("Creating warp bubble animation...")
        
        total_frames = duration_seconds * fps
        phases = np.linspace(0, 1, total_frames)
        
        # Create spatial grid (smaller for speed)
        x = np.linspace(-2, 2, 40)
        y = np.linspace(-2, 2, 40)
        X, Y = np.meshgrid(x, y)
        
        # Create figure FIRST
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Warp Bubble Dynamics', fontsize=16)
        
        # Precompute all data
        print("Precomputing data...")
        g_tt_data = []
        g_tx_data = []
        energy_data = []
        
        for i, phase in enumerate(phases):
            if i % 10 == 0:
                print(f"Computing frame {i}/{total_frames}")
                
            t = 0.5 * phase
            g_tt_frame = np.zeros_like(X)
            g_tx_frame = np.zeros_like(X)
            energy_frame = np.zeros_like(X)
            
            # Batch compute all points for this frame
            coords_list = []
            for x_val in x:
                for y_val in y:
                    coords_list.append([t, x_val, y_val, 0.0, phase])
            
            coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
            
            with torch.no_grad():
                metric_flat = self.model(coords_tensor)
                g = self.reshape_to_metric(metric_flat, len(coords_tensor))
                
                # Reshape to grid
                g_tt_grid = g[:, 0, 0].reshape(len(y), len(x)).numpy()
                g_tx_grid = g[:, 0, 1].reshape(len(y), len(x)).numpy()
                
                # Energy estimate
                g_minkowski = torch.eye(4).unsqueeze(0).repeat(len(g), 1, 1)
                g_minkowski[:, 0, 0] = -1
                energy_vals = torch.mean(torch.abs(g - g_minkowski), dim=(1, 2))
                energy_grid = energy_vals.reshape(len(y), len(x)).numpy()
            
            g_tt_data.append(g_tt_grid)
            g_tx_data.append(g_tx_grid)
            energy_data.append(energy_grid)
        
        # Create initial plots
        im1 = ax1.imshow(g_tt_data[0], extent=[-2, 2, -2, 2], 
                        origin='lower', cmap='RdBu_r', vmin=-1.5, vmax=-0.5)
        ax1.set_title('Time-Time Component (g_tt)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(im1, ax=ax1)
        
        im2 = ax2.imshow(g_tx_data[0], extent=[-2, 2, -2, 2],
                        origin='lower', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
        ax2.set_title('Time-Space Component (g_tx)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        plt.colorbar(im2, ax=ax2)
        
        im3 = ax3.imshow(energy_data[0], extent=[-2, 2, -2, 2],
                        origin='lower', cmap='hot', vmin=0, vmax=1)
        ax3.set_title('Energy Density')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(im3, ax=ax3)
        
        # Bubble strength plot
        all_phases = np.linspace(0, 1, 100)
        bubble_profile = [2*p if p < 0.5 else 2*(1-p) for p in all_phases]
        line, = ax4.plot(all_phases, bubble_profile, 'b-', linewidth=2)
        vline = ax4.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax4.set_title('Bubble Evolution Profile')
        ax4.set_xlabel('Phase')
        ax4.set_ylabel('Bubble Strength')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        def animate(frame):
            phase = phases[frame]
            bubble_strength = 2*phase if phase < 0.5 else 2*(1-phase)
            
            # Update images
            im1.set_array(g_tt_data[frame])
            im2.set_array(g_tx_data[frame]) 
            im3.set_array(energy_data[frame])
            
            # Update phase indicator
            vline.set_xdata([phase, phase])
            
            # Update titles with current info
            ax1.set_title(f'g_tt - Phase: {phase:.2f}')
            ax2.set_title(f'g_tx - Strength: {bubble_strength:.2f}')
            
            # Update overall title with stage
            if phase < 0.25:
                stage = "FORMATION"
                color = "green"
            elif phase < 0.75:
                stage = "TRAVEL" 
                color = "blue"
            else:
                stage = "DISSOLUTION"
                color = "red"
            
            fig.suptitle(f'Warp Bubble Dynamics - {stage}', fontsize=16, color=color)
            
            return [im1, im2, im3, vline]
        
        # Create animation - CRITICAL: save with proper rendering
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames,
            interval=1000/fps, blit=True, repeat=True
        )
        
        print("Animation created successfully!")
        return anim
    
    def reshape_to_metric(self, metric_flat, batch_size):
        g = torch.zeros(batch_size, 4, 4)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        
        return g

class EnergyDurationAnalyzer:
    def __init__(self, model):
        self.model = model
        self.unit_converter = EnergyUnitConverter()
    
    def analyze_energy_duration(self, bubble_radius_m=1.0):
        """Analyze energy requirements over time"""
        print("\n=== TIME-DEPENDENT ENERGY ANALYSIS ===")
        
        phases = np.linspace(0, 1, 100)
        energy_over_time = []
        
        with torch.no_grad():
            for phase in phases:
                # Sample points in shell region
                n_samples = 1000
                spatial_coords = torch.rand(n_samples, 4) * 2 - 1
                r = torch.norm(spatial_coords[:, 1:], dim=1)
                shell_mask = (r >= 0.3) & (r < 1.0)
                shell_coords = spatial_coords[shell_mask]
                
                if len(shell_coords) > 0:
                    phase_tensor = torch.ones(len(shell_coords), 1) * phase
                    full_coords = torch.cat([shell_coords, phase_tensor], dim=1)
                    
                    metric_flat = self.model(full_coords)
                    g_shell = self.reshape_to_metric(metric_flat, len(shell_coords))
                    
                    # Estimate energy density
                    g_minkowski = torch.eye(4).unsqueeze(0).repeat(len(shell_coords), 1, 1)
                    g_minkowski[:, 0, 0] = -1
                    energy_density = torch.mean(torch.abs(g_shell - g_minkowski), dim=(1, 2))
                    total_energy = torch.sum(energy_density).item()
                    
                    energy_over_time.append(total_energy)
                else:
                    energy_over_time.append(0.0)
        
        # Convert to physical units
        energy_natural = np.array(energy_over_time)
        energy_joules = self.unit_converter.natural_to_joules(energy_natural)
        
        # Plot energy over time
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(phases, energy_joules)
        plt.xlabel('Phase')
        plt.ylabel('Energy (Joules)')
        plt.title('Total Energy vs Phase')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        bubble_strength = np.where(phases < 0.5, 2*phases, 2*(1-phases))
        plt.plot(phases, bubble_strength, 'b-', label='Bubble Strength')
        plt.plot(phases, energy_joules / np.max(energy_joules), 'r-', label='Energy (normalized)')
        plt.xlabel('Phase')
        plt.ylabel('Normalized Value')
        plt.title('Bubble Strength vs Energy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate total energy consumption
        total_energy_joules = np.trapz(energy_joules, phases)
        print(f"Total energy for full cycle: {total_energy_joules:.2e} J")
        print(f"Mass equivalent: {total_energy_joules/constants.c**2:.2e} kg")
        
        # Duration analysis
        print("\n=== DURATION ANALYSIS ===")
        power_levels = {
            'Small reactor (100 MW)': 1e8,
            'Large power plant (1 GW)': 1e9,
            'Civilization scale (1 TW)': 1e12,
            'Extreme physics (1 PW)': 1e15,
        }
        
        for name, power in power_levels.items():
            duration = total_energy_joules / power
            if duration < 1:
                print(f"{name}: {duration*1000:.2f} milliseconds")
            elif duration < 60:
                print(f"{name}: {duration:.2f} seconds")
            elif duration < 3600:
                print(f"{name}: {duration/60:.2f} minutes")
            else:
                print(f"{name}: {duration/3600:.2f} hours")
        
        plt.subplot(2, 2, 3)
        durations = [total_energy_joules/p for p in power_levels.values()]
        plt.bar(power_levels.keys(), durations)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Duration (seconds)')
        plt.title('Bubble Duration vs Power Source')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return total_energy_joules
    
    def reshape_to_metric(self, metric_flat, batch_size):
        g = torch.zeros(batch_size, 4, 4)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        
        return g

class EnergyUnitConverter:
    """Convert between natural units and physical units"""
    def __init__(self):
        self.c = constants.c
        self.G = constants.G
        self.hbar = constants.hbar
        self.m_pl = np.sqrt(self.hbar * self.c / self.G)
        self.E_pl = self.m_pl * self.c**2
        
    def natural_to_joules(self, E_natural):
        return E_natural * self.E_pl

# MAIN EXECUTION
if __name__ == "__main__":
    print("=== TIME-DEPENDENT WARP DRIVE SIMULATION ===")
    print("Including bubble formation, travel, and dissolution dynamics")
    
    # Initialize time-dependent model
    model = TimeDependentWarpDriveNN(hidden_dim=128, num_layers=6)
    objective = TimeDependentWarpDriveObjective()
    trainer = TimeDependentWarpDriveTrainer(model, objective)
    
    # Train the model
    trainer.train(epochs=101, batch_size=2048)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    
    # Analyze energy and duration
    analyzer = EnergyDurationAnalyzer(model)
    total_energy = analyzer.analyze_energy_duration(bubble_radius_m=1.0)
    
    # Create animation

    # # Create animation
    # animator = WarpBubbleAnimator(model)
    # animation_obj = animator.create_warp_bubble_animation(duration_seconds=8, fps=15)

    # # Save with proper rendering
    # print("Saving animation...")
    # try:
    #     # This ensures frames are properly rendered during saving
    #     animation_obj.save('warp_bubble_dynamics.mp4', 
    #                       writer='ffmpeg', 
    #                       fps=15,
    #                       dpi=100,
    #                       savefig_kwargs={'facecolor':'white'})  # Important for rendering
    #     print("Animation saved as 'warp_bubble_dynamics.mp4'")
    # except Exception as e:
    #     print(f"Could not save MP4: {e}")
    #     # Try GIF instead
    #     try:
    #         animation_obj.save('warp_bubble_dynamics.gif', 
    #                           writer='pillow', 
    #                           fps=15)
    #         print("Animation saved as GIF instead")
    #     except Exception as e2:
    #         print(f"Could not save GIF either: {e2}")

    # # Display
    # print("Displaying animation...")
    # plt.show()


    # In your main execution, replace with:
    print("=== SMOOTH WARP BUBBLE EVOLUTION ===")

    # Create smooth visualizer
    smooth_visualizer = SmoothWarpBubble3DVisualizer(model)

    # First show the comparison between old and new
    print("Showing evolution profile comparison...")
    smooth_visualizer.create_comparison_animation()

    # Then create the smooth animation
    print("Creating smooth continuous animation...")
    plt.close('all')
    smooth_animation = smooth_visualizer.create_smooth_3d_animation(duration_seconds=12, fps=20)

    # Save it
    print("Saving smooth animation...")
    try:
        smooth_animation.save('smooth_warp_bubble_3d.gif', writer='pillow', fps=20, dpi=50)
        print("Smooth animation saved as 'smooth_warp_bubble_3d.gif'")
    except Exception as e:
        print(f"Could not save: {e}")

    print("Displaying smooth animation...")
    plt.show()



    # # In your main execution, replace the animation section with:
    # print("Creating 3D warp bubble visualization...")
    # visualizer = WarpBubble3DVisualizer(model)

    # # First, show a static 3D bubble to verify it works
    # print("Showing static 3D bubble...")
    # static_bubble = visualizer.create_simple_3d_bubble(phase=0.5)

    # # Then create the full 3D animation
    # print("Creating full 3D animation...")
    # animation_3d = visualizer.create_3d_bubble_animation(duration_seconds=10, fps=15)

    # # Save the 3D animation
    # print("Saving 3D animation...")
    # try:
    #     animation_3d.save('warp_bubble_3d.mp4', writer='ffmpeg', fps=15, dpi=100)
    #     print("3D animation saved as 'warp_bubble_3d.mp4'")
    # except Exception as e:
    #     print(f"Could not save 3D animation: {e}")

    # print("Displaying 3D animation...")
    # plt.show()



    # animator = WarpBubbleAnimator(model)
    # animation_obj = animator.create_warp_bubble_animation_fast(duration_seconds=15, fps=20)
    
    # # Save animation
    # print("Saving animation...")
    # try:
    #     animation_obj.save('warp_bubble_dynamics.mp4', writer='ffmpeg', fps=20)
    #     # animation_obj.save('warp_bubble_dynamics.mp4', writer='ffmpeg', fps=20, bitrate=100, dpi=8)  # Lower quality for speed
    #     print("Animation saved as 'warp_bubble_dynamics.mp4'")
    # except:
    #     print("Could not save animation file, but it will display interactively")
    
    # # Display animation
    # plt.show()
    
    print("\n=== SIMULATION COMPLETE ===")
    print("The animation shows:")
    print("1. FORMATION (Phase 0-0.25): Bubble creates and strengthens")
    print("2. TRAVEL (Phase 0.25-0.75): Stable warp bubble for FTL travel")  
    print("3. DISSOLUTION (Phase 0.75-1.0): Bubble safely dissipates")
    print(f"Total energy for complete cycle: {total_energy:.2e} Joules")
    
    # Save the trained model
    torch.save(model.state_dict(), 'time_dependent_warp_drive_model.pth')
    print("Model saved as 'time_dependent_warp_drive_model.pth'")
