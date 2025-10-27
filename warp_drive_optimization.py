import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from scipy import constants
from scipy.integrate import solve_ivp
import time

# ============================================================================
# 1. NEURAL NETWORK ARCHITECTURE
# ============================================================================

class TimeDependentWarpDriveNN(nn.Module):
    def __init__(self, hidden_dim=1024, num_layers=1000):
        super(TimeDependentWarpDriveNN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(5, hidden_dim))  # Input: (t, x, y, z, phase)
        # layers.append(nn.Tanh())
        layers.append(nn.ELU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            # layers.append(nn.Tanh())
            layers.append(nn.ELU())
            
        layers.append(nn.Linear(hidden_dim, 10))  # Output: 10 metric components
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, coordinates):
        # return self.network(coordinates)
        x = coordinates
        x = self.network[0](x)
        x = self.network[1](x)
        skip = x
        for i in range(len(self.network)-3):
            x = self.network[i+2](x) + skip
        x = self.network[-1](x)
        return x

# ============================================================================
# 1. EINSTEIN-CONSTRAINED NEURAL NETWORK ARCHITECTURE
# ============================================================================

class EinsteinConstrainedWarpDriveNN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=6):
        super().__init__()
        # Network learns a scalar potential Φ(t,x,y,z,phase)
        self.potential_network = nn.Sequential(
            nn.Linear(5, hidden_dim), 
            nn.Tanh(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Tanh()) 
              for _ in range(num_layers-2)],
            nn.Linear(hidden_dim, 1)  # Output: scalar potential Φ
        )
    
    def forward(self, coordinates):
        # For validation/inference: use a simpler approach without gradients
        if not coordinates.requires_grad:
            # Simple forward pass without derivative computation
            phi = self.potential_network(coordinates)
            
            # Use finite differences to approximate derivatives for validation
            eps = 1e-4
            batch_size = coordinates.shape[0]
            
            # Approximate ∂Φ/∂x using finite differences
            coords_plus_x = coordinates.clone()
            coords_plus_x[:, 1] += eps  # perturb x coordinate
            phi_plus_x = self.potential_network(coords_plus_x)
            dphi_dx = (phi_plus_x - phi) / eps
            
            # Similarly for other derivatives...
            coords_plus_y = coordinates.clone()
            coords_plus_y[:, 2] += eps
            phi_plus_y = self.potential_network(coords_plus_y)
            dphi_dy = (phi_plus_y - phi) / eps
            
            coords_plus_z = coordinates.clone()
            coords_plus_z[:, 3] += eps  
            phi_plus_z = self.potential_network(coords_plus_z)
            dphi_dz = (phi_plus_z - phi) / eps
            
            phi = phi.squeeze()
            dphi_dx = dphi_dx.squeeze()
            dphi_dy = dphi_dy.squeeze()
            dphi_dz = dphi_dz.squeeze()
            
        else:
            # For training: use exact gradients
            coordinates = coordinates.clone().requires_grad_(True)
            phi = self.potential_network(coordinates)
            
            gradients = torch.autograd.grad(
                phi, coordinates, 
                grad_outputs=torch.ones_like(phi),
                create_graph=True, 
                retain_graph=True
            )[0]
            
            phi = phi.squeeze()
            dphi_dx = gradients[:, 1]
            dphi_dy = gradients[:, 2]  
            dphi_dz = gradients[:, 3]
        
        # Construct metric (same for both cases)
        batch_size = coordinates.shape[0]
        g = torch.zeros(batch_size, 4, 4, device=coordinates.device)
        
        g[:, 0, 0] = -1.0 + 0.1 * phi**2
        g[:, 0, 1] = -dphi_dx * 2.0
        g[:, 1, 0] = -dphi_dx * 2.0
        g[:, 0, 2] = -dphi_dy * 0.5
        g[:, 2, 0] = -dphi_dy * 0.5
        g[:, 0, 3] = -dphi_dz * 0.5
        g[:, 3, 0] = -dphi_dz * 0.5
        
        g[:, 1, 1] = 1.0 + 0.05 * dphi_dx**2
        g[:, 2, 2] = 1.0 + 0.05 * dphi_dy**2  
        g[:, 3, 3] = 1.0 + 0.05 * dphi_dz**2
        
        g[:, 1, 2] = 0.02 * dphi_dx * dphi_dy
        g[:, 2, 1] = 0.02 * dphi_dx * dphi_dy
        g[:, 1, 3] = 0.02 * dphi_dx * dphi_dz
        g[:, 3, 1] = 0.02 * dphi_dx * dphi_dz
        g[:, 2, 3] = 0.02 * dphi_dy * dphi_dz  
        g[:, 3, 2] = 0.02 * dphi_dy * dphi_dz
        
        # Flatten to 10 components
        metric_flat = torch.zeros(batch_size, 10, device=coordinates.device)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        
        for idx, (i, j) in enumerate(indices):
            metric_flat[:, idx] = g[:, i, j]
        
        return metric_flat

# Then in your main execution, replace:
# model = TimeDependentWarpDriveNN(hidden_dim=128, num_layers=6)
# With:
# model = EinsteinConstrainedWarpDriveNN(hidden_dim=128, num_layers=6)

# ============================================================================
# 2. PROPULSION-OPTIMIZED OBJECTIVE FUNCTION
# ============================================================================

class PropulsiveWarpDriveObjective:
    def __init__(self, weights=None):
        self.weights = weights or {
            'einstein': 1.0,
            'warp_bubble': 1.0,
            'energy_violation': 1.0,
            'energy_minimization': 0.0,
            'causality': 1.0,
            'boundary': 1.0,
            'regularity': 1.0,
            'time_evolution': 1.0,
            'bubble_dynamics': 1.0,
            'propulsion': 1.0,  # Strong weight for propulsion!
            'negative_energy_forward': 1.0,
            'positive_energy_aft': 1.0,
            'mass_energy': 0.0,
            'dynamic_evolution': 1.0,
            'boundary_phases': 1.0
        }
    
    def __call__(self, model, coordinates):
        total_loss = 0.0
        loss_components = {}
        
        spatial_coords = coordinates[:, :4]
        phase = coordinates[:, 4:5]
        
        # Physics constraints
        loss_einstein = self.compute_einstein_loss(model, spatial_coords, phase)
        total_loss += self.weights['einstein'] * loss_einstein
        loss_components['einstein'] = loss_einstein.item()
        
        loss_bubble = self.compute_warp_bubble_loss(model, spatial_coords, phase)
        total_loss += self.weights['warp_bubble'] * loss_bubble
        loss_components['bubble'] = loss_bubble.item()
        
        loss_energy_violation = self.compute_energy_condition_loss(model, spatial_coords, phase)
        total_loss += self.weights['energy_violation'] * loss_energy_violation
        loss_components['energy_violation'] = loss_energy_violation.item()
        
        loss_energy_minimization = self.compute_energy_minimization_loss(model, spatial_coords, phase)
        total_loss += self.weights['energy_minimization'] * loss_energy_minimization
        loss_components['energy_minimization'] = loss_energy_minimization.item()
        
        loss_causality = self.compute_causality_loss(model, spatial_coords, phase)
        total_loss += self.weights['causality'] * loss_causality
        loss_components['causality'] = loss_causality.item()
        
        loss_boundary = self.compute_boundary_loss(model, spatial_coords, phase)
        total_loss += self.weights['boundary'] * loss_boundary
        loss_components['boundary'] = loss_boundary.item()
        
        loss_regularity = self.compute_regularity_loss(model, spatial_coords, phase)
        total_loss += self.weights['regularity'] * loss_regularity
        loss_components['regularity'] = loss_regularity.item()
        
        loss_time_evolution = self.compute_time_evolution_loss(model, spatial_coords, phase)
        total_loss += self.weights['time_evolution'] * loss_time_evolution
        loss_components['time_evolution'] = loss_time_evolution.item()
        
        loss_bubble_dynamics = self.compute_bubble_dynamics_loss(model, spatial_coords, phase)
        total_loss += self.weights['bubble_dynamics'] * loss_bubble_dynamics
        loss_components['bubble_dynamics'] = loss_bubble_dynamics.item()
        
        # PROPULSION CONSTRAINTS
        loss_propulsion = self.compute_propulsion_loss(model, spatial_coords, phase)
        total_loss += self.weights['propulsion'] * loss_propulsion
        loss_components['propulsion'] = loss_propulsion.item()
        
        loss_neg_energy_forward = self.compute_negative_energy_forward_loss(model, spatial_coords, phase)
        total_loss += self.weights['negative_energy_forward'] * loss_neg_energy_forward
        loss_components['negative_energy_forward'] = loss_neg_energy_forward.item()
        
        loss_pos_energy_aft = self.compute_positive_energy_aft_loss(model, spatial_coords, phase)
        total_loss += self.weights['positive_energy_aft'] * loss_pos_energy_aft
        loss_components['positive_energy_aft'] = loss_pos_energy_aft.item()

        loss_mass_energy = self.compute_mass_energy_loss(model, spatial_coords, phase)
        total_loss += self.weights['mass_energy'] * loss_mass_energy
        loss_components['mass_energy'] = loss_mass_energy.item()

        loss_boundary_phases = self.compute_boundary_phase_loss(model, spatial_coords, phase)
        total_loss += self.weights['boundary_phases'] * loss_boundary_phases
        loss_components['boundary_phases'] = loss_boundary_phases.item()

        loss_dynamic_evolution = self.compute_dynamic_evolution_loss(model, spatial_coords, phase)
        total_loss += self.weights['dynamic_evolution'] * loss_dynamic_evolution  
        loss_components['dynamic_evolution'] = loss_dynamic_evolution.item()
        
        return total_loss, loss_components


    def compute_dynamic_evolution_loss(self, model, spatial_coords, phase):
        """FORCE the bubble to actually form and dissolve by comparing different phases"""
        batch_size = spatial_coords.shape[0]
        
        # Sample the SAME spatial points at DIFFERENT phases
        spatial_samples = []
        for _ in range(batch_size // 8):
            t = torch.rand(1) * 0.5
            x = torch.rand(1) * 2 - 1  # Focus on bubble region
            y = torch.rand(1) * 2 - 1
            z = torch.rand(1) * 2 - 1
            spatial_samples.append(torch.tensor([t, x, y, z]))
        
        if not spatial_samples:
            return torch.tensor(0.0, device=spatial_coords.device)
        
        spatial_samples = torch.stack(spatial_samples)
        
        # Create coordinates at CRITICAL phases
        critical_phases = []
        
        # Phase 0.0 - should be FLAT
        for coords in spatial_samples:
            critical_phases.append(torch.cat([coords, torch.tensor([0.0])]))
        
        # Phase 0.5 - should be MAXIMUM warp
        for coords in spatial_samples:  
            critical_phases.append(torch.cat([coords, torch.tensor([0.5])]))
        
        # Phase 1.0 - should be FLAT again
        for coords in spatial_samples:
            critical_phases.append(torch.cat([coords, torch.tensor([1.0])]))
        
        critical_coords = torch.stack(critical_phases)
        
        # Get metrics at all critical phases
        metric_flat = model(critical_coords)
        g = self.reshape_to_metric(metric_flat, len(critical_coords))
        
        n_samples = len(spatial_samples)
        g_phase_0 = g[:n_samples]           # Start: phase 0.0
        g_phase_max = g[n_samples:2*n_samples]  # Middle: phase 0.5  
        g_phase_1 = g[2*n_samples:]         # End: phase 1.0
        
        g_minkowski = self.get_minkowski_metric(n_samples, spatial_coords.device)
        
        # STRONG CONSTRAINTS:
        loss = 0.0
        
        # 1. Start and end MUST be Minkowski (flat)
        loss += torch.mean((g_phase_0 - g_minkowski) ** 2) * 50.0
        loss += torch.mean((g_phase_1 - g_minkowski) ** 2) * 50.0
        
        # 2. Middle MUST be DIFFERENT from Minkowski (warped)
        middle_deviation = torch.mean(torch.abs(g_phase_max - g_minkowski))
        loss += torch.exp(-middle_deviation * 10.0) * 20.0  # Heavy penalty if middle is flat
        
        # 3. Middle MUST be DIFFERENT from start/end
        loss += torch.exp(-torch.mean(torch.abs(g_phase_max - g_phase_0)) * 10.0) * 10.0
        loss += torch.exp(-torch.mean(torch.abs(g_phase_max - g_phase_1)) * 10.0) * 10.0
        
        return loss

    def compute_boundary_phase_loss(self, model, spatial_coords, phase):
        """Force metric to be exactly Minkowski at phase=0 and phase=1"""
        batch_size = spatial_coords.shape[0]
        
        # Sample points specifically at boundary phases
        boundary_coords = []
        
        # Phase 0.0 (start - flat spacetime)
        for _ in range(batch_size // 16):
            t = torch.rand(1) * 0.5
            x = torch.rand(1) * 4 - 2
            y = torch.rand(1) * 4 - 2  
            z = torch.rand(1) * 4 - 2
            boundary_coords.append(torch.tensor([t, x, y, z, 0.0]))
        
        # Phase 1.0 (end - flat spacetime)
        for _ in range(batch_size // 16):
            t = torch.rand(1) * 0.5
            x = torch.rand(1) * 4 - 2
            y = torch.rand(1) * 4 - 2
            z = torch.rand(1) * 4 - 2
            boundary_coords.append(torch.tensor([t, x, y, z, 1.0]))
        
        if not boundary_coords:
            return torch.tensor(0.0, device=spatial_coords.device)
        
        boundary_coords = torch.stack(boundary_coords)
        
        # Get metrics at boundary phases
        metric_flat = model(boundary_coords)
        g = self.reshape_to_metric(metric_flat, len(boundary_coords))
        
        # Target: perfect Minkowski metric
        g_minkowski = self.get_minkowski_metric(len(boundary_coords), spatial_coords.device)
        
        # STRONG penalty for deviation from Minkowski at boundaries
        boundary_loss = torch.mean((g - g_minkowski) ** 2) * 100.0  # Heavy weight
        
        return boundary_loss

    # Add this method to your class:
    def compute_mass_energy_loss(self, model, spatial_coords, phase):
        """Ensure the warp drive has non-zero mass-energy"""
        batch_size = spatial_coords.shape[0]
        mass_coords = self.sample_mass_estimation_coordinates(batch_size // 8)
        mass_coords = torch.cat([mass_coords, torch.ones(len(mass_coords), 1) * 0.5], dim=1)
        
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        all_coords = torch.cat([full_coords, mass_coords], dim=0)
        
        metric_flat = model(all_coords)
        g = self.reshape_to_metric(metric_flat, len(all_coords))
        mass_metrics = g[-len(mass_coords):]
        
        g_minkowski = self.get_minkowski_metric(len(mass_metrics), spatial_coords.device)
        metric_deviation = torch.mean(torch.abs(mass_metrics - g_minkowski), dim=(1, 2))
        total_energy = torch.sum(metric_deviation)
        
        target_mass = 1.0
        mass_loss = (total_energy - target_mass) ** 2
        negative_mass_penalty = torch.nn.functional.softplus(-total_energy * 10.0)
        
        return mass_loss + negative_mass_penalty
    
    # Add this method too:
    def sample_mass_estimation_coordinates(self, n_samples):
        """Sample coordinates for mass-energy estimation"""
        coords = []
        for _ in range(n_samples):
            t = torch.rand(1) * 0.5
            r = torch.rand(1) * 2.0
            theta = torch.rand(1) * 2 * np.pi
            phi = torch.rand(1) * np.pi
            x = r * torch.sin(phi) * torch.cos(theta)
            y = r * torch.sin(phi) * torch.sin(theta)
            z = r * torch.cos(phi)
            coords.append(torch.tensor([t, x, y, z]))
        return torch.stack(coords)
    
    def compute_einstein_loss(self, model, spatial_coords, phase):
        batch_size = spatial_coords.shape[0]
        spatial_coords = spatial_coords.clone().requires_grad_(True)
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        
        metric_flat = model(full_coords)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
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
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        
        r = torch.norm(spatial_coords[:, 1:], dim=1)
        metric_flat = model(full_coords)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        loss_bubble = 0.0
        g_minkowski = self.get_minkowski_metric(batch_size, spatial_coords.device)
        bubble_strength = self.bubble_evolution_profile(phase)
        
        inside_mask = r < 0.3
        shell_mask = (r >= 0.3) & (r < 1.0)
        outside_mask = r >= 1.0
        
        if torch.any(inside_mask):
            g_inside = g[inside_mask]
            target_inside = g_minkowski[inside_mask] + (1 - bubble_strength[inside_mask].unsqueeze(-1).unsqueeze(-1)) * 0.1
            loss_inside = torch.mean((g_inside - target_inside)**2)
            loss_bubble += loss_inside
        
        if torch.any(shell_mask):
            g_shell = g[shell_mask]
            shell_strength = bubble_strength[shell_mask]
            target_warp = 0.5 * shell_strength.unsqueeze(-1).unsqueeze(-1)
            loss_shell = torch.mean((g_shell - (g_minkowski[shell_mask] + target_warp))**2)
            loss_bubble += loss_shell
        
        if torch.any(outside_mask):
            g_outside = g[outside_mask]
            loss_outside = torch.mean((g_outside - g_minkowski[outside_mask])**2)
            loss_bubble += loss_outside
        
        return loss_bubble
    
    def compute_energy_condition_loss(self, model, spatial_coords, phase):
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        
        r = torch.norm(spatial_coords[:, 1:], dim=1)
        shell_mask = (r >= 0.3) & (r < 1.0)
        
        if not torch.any(shell_mask):
            return torch.tensor(0.0, device=spatial_coords.device)
        
        metric_flat = model(full_coords[shell_mask])
        g_shell = self.reshape_to_metric(metric_flat, torch.sum(shell_mask))
        g_tt_shell = g_shell[:, 0, 0]
        target_g_tt = -0.7
        return torch.mean((g_tt_shell - target_g_tt)**2)
    
    def compute_energy_minimization_loss(self, model, spatial_coords, phase):
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        full_coords.requires_grad_(True)
        
        r = torch.norm(spatial_coords[:, 1:], dim=1)
        shell_mask = (r >= 0.3) & (r < 1.0)
        
        if not torch.any(shell_mask):
            return torch.tensor(0.0, device=spatial_coords.device)
        
        metric_flat = model(full_coords[shell_mask])
        g_shell = self.reshape_to_metric(metric_flat, torch.sum(shell_mask))
        
        g_minkowski = self.get_minkowski_metric(len(g_shell), spatial_coords.device)
        metric_deviation = g_shell - g_minkowski
        bubble_strength = self.bubble_evolution_profile(phase[shell_mask])
        
        energy_density_estimate = (
            torch.abs(metric_deviation[:, 0, 0]) +
            torch.abs(metric_deviation[:, 0, 1]) +
            torch.abs(metric_deviation[:, 0, 2]) +
            torch.abs(metric_deviation[:, 0, 3])
        ) * bubble_strength
        
        loss_energy_minimization = torch.mean(energy_density_estimate**2)
        
        loss_gradients = 0.0
        for i in range(4):
            for j in range(4):
                g_ij = g_shell[:, i, j]
                grad_g = torch.autograd.grad(g_ij, full_coords, grad_outputs=torch.ones_like(g_ij),
                                           create_graph=True, retain_graph=True)[0]
                loss_gradients += torch.mean(grad_g**2)
        
        return loss_energy_minimization + 0.1 * loss_gradients
    
    def compute_causality_loss(self, model, spatial_coords, phase):
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        
        metric_flat = model(full_coords)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        det_g = torch.det(g)
        loss_det = torch.mean(torch.relu(det_g + 1e-6))
        g_tt = g[:, 0, 0]
        loss_g_tt = torch.mean(torch.relu(g_tt + 1e-6))
        
        return loss_det + loss_g_tt
    
    def compute_boundary_loss(self, model, spatial_coords, phase):
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
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        full_coords.requires_grad_(True)
        
        metric_flat = model(full_coords)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        loss_magnitude = torch.mean(g**2)
        loss_smoothness = 0.0
        for i in range(4):
            for j in range(4):
                g_ij = g[:, i, j]
                grad_g = torch.autograd.grad(g_ij, full_coords, grad_outputs=torch.ones_like(g_ij),
                                           create_graph=True, retain_graph=True)[0]
                loss_smoothness += torch.mean(grad_g**2)
        
        return 0.1 * loss_magnitude + 0.01 * loss_smoothness
    
    def compute_time_evolution_loss(self, model, spatial_coords, phase):
        batch_size = spatial_coords.shape[0]
        spatial_coords = spatial_coords.clone().requires_grad_(True)
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        
        metric_flat = model(full_coords)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        loss_time_derivatives = 0.0
        for i in range(4):
            for j in range(4):
                g_ij = g[:, i, j]
                grad_g = torch.autograd.grad(g_ij, spatial_coords, grad_outputs=torch.ones_like(g_ij),
                                           create_graph=True, retain_graph=True)[0]
                time_deriv = grad_g[:, 0]
                loss_time_derivatives += torch.mean(time_deriv**2)
        
        return loss_time_derivatives * 0.1
    
    def compute_bubble_dynamics_loss(self, model, spatial_coords, phase):
        batch_size = spatial_coords.shape[0]
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        
        r = torch.norm(spatial_coords[:, 1:], dim=1)
        shell_mask = (r >= 0.3) & (r < 1.0)
        
        if not torch.any(shell_mask):
            return torch.tensor(0.0, device=spatial_coords.device)
        
        metric_flat = model(full_coords[shell_mask])
        g_shell = self.reshape_to_metric(metric_flat, torch.sum(shell_mask))
        bubble_strength = self.bubble_evolution_profile(phase[shell_mask])
        
        g_minkowski = self.get_minkowski_metric(len(g_shell), spatial_coords.device)
        actual_warping = torch.mean(torch.abs(g_shell - g_minkowski), dim=(1, 2))
        target_warping = 0.3 * bubble_strength
        
        return torch.mean((actual_warping - target_warping)**2)

    # def compute_propulsion_loss(self, model, spatial_coords, phase):
    #     """Elegant propulsion loss with numerical stability"""
    #     batch_size = spatial_coords.shape[0]
        
    #     propulsion_coords = self.sample_propulsion_coordinates(batch_size // 4)
    #     propulsion_coords = torch.cat([propulsion_coords, torch.ones(len(propulsion_coords), 1) * 0.5], dim=1)
        
    #     full_coords = torch.cat([spatial_coords, phase], dim=1)
    #     all_coords = torch.cat([full_coords, propulsion_coords], dim=0)
        
    #     metric_flat = model(all_coords)
    #     g = self.reshape_to_metric(metric_flat, len(all_coords))
    #     propulsion_metrics = g[-len(propulsion_coords):]
        
    #     g_tt = propulsion_metrics[:, 0, 0]
    #     g_tx = propulsion_metrics[:, 0, 1]
        
    #     # ELEGANT VELOCITY CALCULATION: Handle all cases smoothly
    #     # Use softplus to ensure positive denominator and smooth gradients
    #     denominator = torch.sqrt(torch.nn.functional.softplus(-g_tt, beta=10.0) + 1e-8)
    #     velocity = -g_tx / denominator
        
    #     target_velocity = 2.0  # Warp 2
        
    #     # ELEGANT LOSS COMPONENTS:
    #     velocity_loss = torch.mean((velocity - target_velocity) ** 2)
    #     wrong_direction_penalty = torch.mean(torch.nn.functional.softplus(-velocity, beta=5.0))
    #     weak_metric_penalty = torch.mean(torch.exp(-torch.abs(g_tx) * 5.0))
        
    #     # ELEGANT COMBINATION:
    #     total_propulsion_loss = velocity_loss + wrong_direction_penalty + weak_metric_penalty
        
    #     return total_propulsion_loss


    def compute_propulsion_loss(self, model, spatial_coords, phase):
        """Elegant propulsion loss: faster is better, period"""
        batch_size = spatial_coords.shape[0]
        
        propulsion_coords = self.sample_propulsion_coordinates(batch_size // 4)
        propulsion_coords = torch.cat([propulsion_coords, torch.ones(len(propulsion_coords), 1) * 0.5], dim=1)
        
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        all_coords = torch.cat([full_coords, propulsion_coords], dim=0)
        
        metric_flat = model(all_coords)
        g = self.reshape_to_metric(metric_flat, len(all_coords))
        propulsion_metrics = g[-len(propulsion_coords):]
        
        g_tt = propulsion_metrics[:, 0, 0]
        g_tx = propulsion_metrics[:, 0, 1]
        
        # Elegant velocity calculation
        denominator = torch.sqrt(torch.nn.functional.softplus(-g_tt, beta=10.0) + 1e-8)
        velocity = -g_tx / denominator
        
        # ELEGANT LOSS: FASTER IS BETTER
        # Use negative exponential to reward high velocities
        propulsion_loss = torch.mean(torch.exp(-velocity))  # Minimize this = maximize velocity
        # propulsion_loss = torch.mean(-velocity)
        
        # Penalize negative velocities heavily
        wrong_direction_penalty = torch.mean(torch.nn.functional.softplus(-velocity * 10.0))
        
        # Encourage strong warp effects (large |g_tx|)
        warp_strength_penalty = torch.mean(torch.exp(-torch.abs(g_tx) * 5.0))
        
        return propulsion_loss #+ wrong_direction_penalty #+ warp_strength_penalty
    
    def compute_negative_energy_forward_loss(self, model, spatial_coords, phase):
        batch_size = spatial_coords.shape[0]
        forward_coords = self.sample_forward_region_coordinates(batch_size // 8)
        forward_coords = torch.cat([forward_coords, torch.ones(batch_size // 8, 1) * 0.5], dim=1)
        
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        all_coords = torch.cat([full_coords, forward_coords], dim=0)
        
        metric_flat = model(all_coords)
        g = self.reshape_to_metric(metric_flat, len(all_coords))
        forward_metrics = g[-batch_size//8:]
        
        loss_negative = 0.0
        for i in range(len(forward_metrics)):
            g_tt = forward_metrics[i, 0, 0].item()
            energy_density = -(g_tt + 1.0)
            loss_negative += (energy_density - (-0.2)) ** 2
        
        return torch.tensor(loss_negative / len(forward_metrics), device=spatial_coords.device)
    
    def compute_positive_energy_aft_loss(self, model, spatial_coords, phase):
        batch_size = spatial_coords.shape[0]
        aft_coords = self.sample_aft_region_coordinates(batch_size // 8)
        aft_coords = torch.cat([aft_coords, torch.ones(batch_size // 8, 1) * 0.5], dim=1)
        
        full_coords = torch.cat([spatial_coords, phase], dim=1)
        all_coords = torch.cat([full_coords, aft_coords], dim=0)
        
        metric_flat = model(all_coords)
        g = self.reshape_to_metric(metric_flat, len(all_coords))
        aft_metrics = g[-batch_size//8:]
        
        loss_positive = 0.0
        for i in range(len(aft_metrics)):
            g_tt = aft_metrics[i, 0, 0].item()
            energy_density = -(g_tt + 1.0)
            loss_positive += (energy_density - 0.2) ** 2
        
        return torch.tensor(loss_positive / len(aft_metrics), device=spatial_coords.device)
    
    def sample_propulsion_coordinates(self, n_samples):
        coords = []
        for _ in range(n_samples // 2):
            t, x = torch.rand(1) * 0.5, torch.rand(1) * 1.5 + 0.3
            y, z = torch.rand(1) * 0.5 - 0.25, torch.rand(1) * 0.5 - 0.25
            coords.append(torch.tensor([t, x, y, z]))
        for _ in range(n_samples // 2):
            t, x = torch.rand(1) * 0.5, torch.rand(1) * 1.5 - 1.8
            y, z = torch.rand(1) * 0.5 - 0.25, torch.rand(1) * 0.5 - 0.25
            coords.append(torch.tensor([t, x, y, z]))
        return torch.stack(coords)
    
    def sample_forward_region_coordinates(self, n_samples):
        coords = []
        for _ in range(n_samples):
            t = torch.rand(1) * 0.5
            x = torch.rand(1) * 1.2 + 0.3
            y = torch.rand(1) * 1.0 - 0.5
            z = torch.rand(1) * 1.0 - 0.5
            coords.append(torch.tensor([t, x, y, z]))
        return torch.stack(coords)
    
    def sample_aft_region_coordinates(self, n_samples):
        coords = []
        for _ in range(n_samples):
            t = torch.rand(1) * 0.5
            x = torch.rand(1) * 1.2 - 1.5
            y = torch.rand(1) * 1.0 - 0.5
            z = torch.rand(1) * 1.0 - 0.5
            coords.append(torch.tensor([t, x, y, z]))
        return torch.stack(coords)
    
    def bubble_evolution_profile(self, phase):
        return torch.where(phase < 0.5, 2 * phase, 2 * (1 - phase))
    
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

# ============================================================================
# 3. PROPULSION-OPTIMIZED TRAINER
# ============================================================================

class PropulsiveWarpDriveTrainer:
    def __init__(self, model, objective):
        self.model = model
        self.objective = objective
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        # self.optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=50)
        self.propulsion_history = {
            'net_force': [], 'forward_velocity': [], 'velocity_gradient': [], 'energy_difference': []
        }
    
    def train_with_propulsion(self, epochs=2000, batch_size=1024, validation_interval=100):
        print("🚀 TRAINING PROPULSIVE WARP DRIVE...")
        
        for epoch in range(epochs):
            coordinates = self.sample_training_points(batch_size)
            total_loss, loss_components = self.objective(self.model, coordinates)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            self.scheduler.step(total_loss)
            
            if epoch % validation_interval == 0:
                self.print_progress(epoch, total_loss, loss_components)
                self.validate_propulsion(epoch)
                self.adapt_propulsion_weights(epoch)
    
    def validate_propulsion(self, epoch):
        propulsion_detected, velocities, energies = self.verify_forward_propulsion(phase=0.5)
        test_points = np.linspace(-1.5, 1.5, 50)
        
        avg_forward_v = np.mean([v for v, x in zip(velocities, test_points) if x > 0.3])
        avg_aft_v = np.mean([v for v, x in zip(velocities, test_points) if x < -0.3])
        velocity_gradient = avg_forward_v - avg_aft_v
        
        avg_forward_e = np.mean([e for e, x in zip(energies, test_points) if x > 0.3])
        avg_aft_e = np.mean([e for e, x in zip(energies, test_points) if x < -0.3])
        energy_difference = avg_forward_e - avg_aft_e
        
        self.propulsion_history['net_force'].append(velocity_gradient)
        self.propulsion_history['forward_velocity'].append(avg_forward_v)
        self.propulsion_history['velocity_gradient'].append(velocity_gradient)
        self.propulsion_history['energy_difference'].append(energy_difference)
        
        print(f"   Propulsion Metrics: V_grad={velocity_gradient:.4f}, E_diff={energy_difference:.4f}")
        
        avg_velocity = np.mean([abs(v) for v in velocities])  # Use absolute velocity
        # In validate_propulsion method, change:
        if avg_velocity > 2.0:  # Warp 2+
            print(f"   ✅ WARP DRIVE: {avg_velocity:.1f}c")
        elif avg_velocity > 1.0:  # FTL
            print(f"   ✅ FTL PROPULSION: {avg_velocity:.1f}c")  
        elif avg_velocity > 0.3:  # Subluminal but fast
            print(f"   ⚠️  SUBLUMINAL: {avg_velocity:.3f}c")
        else:
            print(f"   ❌ NO PROPULSION: {avg_velocity:.3f}c")
    
    def verify_forward_propulsion(self, phase=0.5):
        test_points = np.linspace(-1.5, 1.5, 50)
        velocities, energy_densities = [], []
        t = 0.2 * phase
        
        with torch.no_grad():
            for x in test_points:
                coords = torch.tensor([[t, x, 0, 0, phase]], dtype=torch.float32)
                metric_flat = self.model(coords)
                g = self.reshape_to_metric(metric_flat, 1)
                g_tt, g_tx = g[0, 0, 0].item(), g[0, 0, 1].item()
                
                velocity = -g_tx / np.sqrt(-g_tt) if g_tt < 0 else 0.0
                energy_density = -(g_tt + 1.0)
                
                velocities.append(velocity)
                energy_densities.append(energy_density)
        
        forward_region = test_points > 0.3
        aft_region = test_points < -0.3
        avg_forward_v = np.mean([v for v, x in zip(velocities, test_points) if x > 0.3])
        avg_aft_v = np.mean([v for v, x in zip(velocities, test_points) if x < -0.3])
        avg_forward_e = np.mean([e for e, x in zip(energy_densities, test_points) if x > 0.3])
        avg_aft_e = np.mean([e for e, x in zip(energy_densities, test_points) if x < -0.3])
        
        propulsion_detected = (abs(avg_forward_v - avg_aft_v) > 0.01) and (avg_forward_e < avg_aft_e)
        return propulsion_detected, velocities, energy_densities
    
    def adapt_propulsion_weights(self, epoch):
        if epoch > 500 and epoch % 200 == 0:
            recent_gradient = np.mean(self.propulsion_history['velocity_gradient'][-5:])
            recent_energy_diff = np.mean(self.propulsion_history['energy_difference'][-5:])
            
            if recent_gradient < 0.15:
                self.objective.weights['propulsion'] *= 1.3
                self.objective.weights['negative_energy_forward'] *= 1.2
                self.objective.weights['positive_energy_aft'] *= 1.2
                print(f"   ↑ Increased propulsion weights to {self.objective.weights['propulsion']:.1f}")
            
            if recent_energy_diff > -0.05:
                self.objective.weights['negative_energy_forward'] *= 1.5
                print(f"   ↑ Increased negative energy weight to {self.objective.weights['negative_energy_forward']:.1f}")
    
    def sample_training_points(self, batch_size):
        spatial = torch.rand(batch_size // 2, 4) * 4 - 2
        phase = torch.rand(batch_size // 2, 1)
        regular_coords = torch.cat([spatial, phase], dim=1)
        propulsion_coords = self.sample_propulsion_focused_coordinates(batch_size // 2)
        return torch.cat([regular_coords, propulsion_coords], dim=0)
    
    def sample_propulsion_focused_coordinates(self, n_samples):
        coords = []
        for _ in range(n_samples // 3):
            t, x = torch.rand(1) * 0.5, torch.rand(1) * 1.2 + 0.3
            y, z = torch.rand(1) * 0.8 - 0.4, torch.rand(1) * 0.8 - 0.4
            phase = torch.rand(1)
            coords.append(torch.tensor([t, x, y, z, phase]))
        for _ in range(n_samples // 3):
            t, x = torch.rand(1) * 0.5, torch.rand(1) * 1.2 - 1.5
            y, z = torch.rand(1) * 0.8 - 0.4, torch.rand(1) * 0.8 - 0.4
            phase = torch.rand(1)
            coords.append(torch.tensor([t, x, y, z, phase]))
        for _ in range(n_samples // 3):
            t, r = torch.rand(1) * 0.5, torch.rand(1) * 0.7 + 0.3
            theta = torch.rand(1) * 2 * np.pi
            x, y = r * torch.cos(theta), r * torch.sin(theta) * 0.5
            z, phase = torch.rand(1) * 0.5 - 0.25, torch.rand(1)
            coords.append(torch.tensor([t, x, y, z, phase]))
        return torch.stack(coords)
    
    def print_progress(self, epoch, total_loss, components):
        print(f"\nEpoch {epoch}: Total Loss = {total_loss.item():.6f}")
        print("Physics Losses:")
        for name, value in components.items():
            if any(prop in name for prop in ['propulsion', 'energy_forward', 'energy_aft']):
                print(f"  🚀 {name:25}: {value:.6f}")
            else:
                print(f"  {name:25}: {value:.6f}")
    
    def reshape_to_metric(self, metric_flat, batch_size):
        g = torch.zeros(batch_size, 4, 4)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        return g

# ============================================================================
# 4. SMOOTH 3D VISUALIZATION
# ============================================================================

class SmoothWarpBubble3DVisualizer:
    def __init__(self, model):
        self.model = model
        
    def smooth_bubble_evolution(self, phase, formation_time=0.3, cruise_time=0.4, dissolution_time=0.3):
        total_time = formation_time + cruise_time + dissolution_time
        formation_time /= total_time
        cruise_time /= total_time
        dissolution_time /= total_time
        
        formation_end = formation_time
        cruise_end = formation_time + cruise_time
        
        if phase < formation_end:
            t_norm = phase / formation_end
            return 0.5 * (1 - np.cos(np.pi * t_norm))
        elif phase < cruise_end:
            return 1.0
        else:
            t_norm = (phase - cruise_end) / dissolution_time
            return 0.5 * (1 + np.cos(np.pi * t_norm))
    
    def create_smooth_3d_animation(self, duration_seconds=10, fps=20):
        print("Creating smooth 3D warp bubble animation...")
        
        total_frames = duration_seconds * fps
        phases = np.linspace(0, 1, total_frames)
        x = np.linspace(-2, 2, 40)
        y = np.linspace(-2, 2, 40)
        X, Y = np.meshgrid(x, y)
        
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
                coords_list = []
                for xi in x:
                    for yi in y:
                        coords_list.append([t, xi, yi, 0.0, phase])
                
                coords_tensor = torch.tensor(coords_list, dtype=torch.float32)
                metric_flat = self.model(coords_tensor)
                g = self.reshape_to_metric(metric_flat, len(coords_list))
                
                g_minkowski = torch.eye(4).unsqueeze(0).repeat(len(coords_list), 1, 1)
                g_minkowski[:, 0, 0] = -1
                warp_strength = torch.mean(torch.abs(g - g_minkowski), dim=(1, 2))
                surface_data = warp_strength.reshape(len(y), len(x)).numpy()
                
                r = np.sqrt(X**2 + Y**2)
                inner_transition = 0.5 * (1 + np.tanh(10 * (r - 0.2)))
                outer_transition = 0.5 * (1 - np.tanh(10 * (r - 0.9)))
                shell_profile = inner_transition * outer_transition
                
                surface_height = surface_data * shell_profile * current_strength * 2.0
                background = 0.1 * current_strength * (1 - shell_profile)
                surface_height += background
            
            bubble_surfaces.append(surface_height)
        
        fig = plt.figure(figsize=(14, 10))
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        surf = ax1.plot_surface(X, Y, bubble_surfaces[0], cmap='viridis', alpha=0.85, linewidth=0.5, antialiased=True)
        timeline, = ax2.plot(phases[:1], bubble_strengths[:1], 'b-', linewidth=3, label='Bubble Strength')
        current_point = ax2.scatter([0], [bubble_strengths[0]], color='red', s=80, zorder=5)
        
        radial_distances = np.linspace(0, 2, 50)
        current_radial = self.compute_radial_profile(phases[0], bubble_strengths[0])
        radial_line, = ax3.plot(radial_distances, current_radial, 'r-', linewidth=2)
        
        info_text = ax4.text(0.05, 0.95, '', transform=ax4.transAxes, fontsize=10,
                           verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        ax4.axis('off')
        
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Warp Field Strength')
        ax1.set_title('3D Warp Bubble - FORMATION')
        ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.1); ax2.set_xlabel('Phase'); ax2.set_ylabel('Bubble Strength')
        ax2.set_title('Evolution Timeline'); ax2.grid(True, alpha=0.3); ax2.legend()
        ax3.set_xlabel('Radial Distance'); ax3.set_ylabel('Warp Strength'); ax3.set_title('Radial Profile')
        ax3.grid(True, alpha=0.3); ax3.set_ylim(0, max(current_radial) * 1.2 if max(current_radial) > 0 else 1)
        
        plt.colorbar(surf, ax=ax1, shrink=0.6, label='Warp Strength')
        plt.tight_layout()
        
        def animate(frame):
            phase = phases[frame]
            strength = bubble_strengths[frame]
            
            ax1.clear()
            surf = ax1.plot_surface(X, Y, bubble_surfaces[frame], cmap='viridis', alpha=0.85, linewidth=0.5, antialiased=True)
            timeline.set_data(phases[:frame+1], bubble_strengths[:frame+1])
            current_point.set_offsets([[phase, strength]])
            
            new_radial = self.compute_radial_profile(phase, strength)
            radial_line.set_ydata(new_radial)
            ax3.set_ylim(0, max(new_radial) * 1.2 if max(new_radial) > 0 else 1)
            
            formation_end, cruise_end = 0.3, 0.7
            if phase < formation_end:
                stage, stage_color, progress = "FORMATION", "green", phase / formation_end
            elif phase < cruise_end:
                stage, stage_color, progress = "CRUISE", "blue", (phase - formation_end) / (cruise_end - formation_end)
            else:
                stage, stage_color, progress = "DISSOLUTION", "red", (phase - cruise_end) / (1 - cruise_end)
            
            ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Warp Field Strength')
            ax1.set_title(f'3D Warp Bubble - {stage}\nPhase: {phase:.2f}', color=stage_color)
            
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
        
        anim = animation.FuncAnimation(fig, animate, frames=total_frames, interval=1000/fps, blit=False, repeat=True)
        print("Smooth 3D animation created successfully!")
        return anim
    
    def compute_radial_profile(self, phase, strength):
        radial_distances = np.linspace(0, 2, 50)
        profile = []
        t = 0.2 * phase
        
        with torch.no_grad():
            for r_val in radial_distances:
                coords = torch.tensor([[t, r_val, 0.0, 0.0, phase]], dtype=torch.float32)
                metric_flat = self.model(coords)
                g = self.reshape_to_metric(metric_flat, 1)
                warp_strength = torch.mean(torch.abs(g - torch.eye(4))).item()
                
                inner = 0.5 * (1 + np.tanh(10 * (r_val - 0.2)))
                outer = 0.5 * (1 - np.tanh(10 * (r_val - 0.9)))
                radial_factor = inner * outer
                
                profile.append(warp_strength * radial_factor * strength * 2.0)
        
        return np.array(profile)
    
    def reshape_to_metric(self, metric_flat, batch_size):
        g = torch.zeros(batch_size, 4, 4)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        return g

# ============================================================================
# 5. PROPULSION VERIFICATION AND ANALYSIS
# ============================================================================

class WarpDrivePropulsionVerifier:
    def __init__(self, model):
        self.model = model
        
    def verify_forward_propulsion(self, phase=0.5):
        print("=== PROPULSION VERIFICATION ===")
        test_points = np.linspace(-1.5, 1.5, 50)
        velocities, energy_densities = [], []
        t = 0.2 * phase
        
        with torch.no_grad():
            for x in test_points:
                coords = torch.tensor([[t, x, 0, 0, phase]], dtype=torch.float32)
                metric_flat = self.model(coords)
                g = self.reshape_to_metric(metric_flat, 1)
                g_tt, g_tx = g[0, 0, 0].item(), g[0, 0, 1].item()
                
                velocity = -g_tx / np.sqrt(-g_tt) if g_tt < 0 else 0.0
                energy_density = -(g_tt + 1.0)
                
                velocities.append(velocity)
                energy_densities.append(energy_density)
        
        avg_forward_v = np.mean([v for v, x in zip(velocities, test_points) if x > 0.3])
        avg_aft_v = np.mean([v for v, x in zip(velocities, test_points) if x < -0.3])
        avg_forward_e = np.mean([e for e, x in zip(energy_densities, test_points) if x > 0.3])
        avg_aft_e = np.mean([e for e, x in zip(energy_densities, test_points) if x < -0.3])
        
        print(f"Average forward velocity (x > 0.3): {avg_forward_v:.4f}c")
        print(f"Average aft velocity (x < -0.3): {avg_aft_v:.4f}c")
        print(f"Velocity difference: {avg_forward_v - avg_aft_v:.4f}c")
        print(f"Average forward energy density: {avg_forward_e:.6f}")
        print(f"Average aft energy density: {avg_aft_e:.6f}")
        print(f"Energy difference: {avg_forward_e - avg_aft_e:.6f}")
        
        propulsion_detected = (abs(avg_forward_v - avg_aft_v) > 0.01) and (avg_forward_e < avg_aft_e)
        
        if propulsion_detected:
            print("✅ PROPULSION DETECTED: Classic warp drive signature")
        else:
            print("❌ NO PROPULSION: Insufficient velocity gradient or wrong energy distribution")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.plot(test_points, velocities, 'b-', linewidth=2, label='Coordinate Velocity')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Bubble Center')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_xlabel('X Position'); ax1.set_ylabel('Coordinate Velocity (c)')
        ax1.set_title('Velocity Field Analysis'); ax1.legend(); ax1.grid(True, alpha=0.3)
        
        ax2.plot(test_points, energy_densities, 'r-', linewidth=2, label='Energy Density')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Bubble Center')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xlabel('X Position'); ax2.set_ylabel('Energy Density (arb. units)')
        ax2.set_title('Energy Distribution'); ax2.legend(); ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return propulsion_detected, velocities, energy_densities
    
    def reshape_to_metric(self, metric_flat, batch_size):
        g = torch.zeros(batch_size, 4, 4)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        return g

# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=== PROPULSION-OPTIMIZED WARP DRIVE DISCOVERY ===")
    print("Training neural network to find novel, propulsive warp drive solutions...")
    
    # Initialize model and objective
    # model = TimeDependentWarpDriveNN(hidden_dim=128, num_layers=6)
    model = EinsteinConstrainedWarpDriveNN(hidden_dim=20, num_layers=6)
    propulsion_objective = PropulsiveWarpDriveObjective()
    propulsion_trainer = PropulsiveWarpDriveTrainer(model, propulsion_objective)
    
    # Train with propulsion optimization
    propulsion_trainer.train_with_propulsion(epochs=1000, batch_size=32, validation_interval=1)
    
    print("\n" + "="*60)
    print("PROPULSION-OPTIMIZED TRAINING COMPLETE!")
    print("="*60)
    
    # Final propulsion verification
    print("\n=== FINAL PROPULSION VERIFICATION ===")
    propulsion_verifier = WarpDrivePropulsionVerifier(model)
    propulsion_detected, velocities, energies = propulsion_verifier.verify_forward_propulsion(phase=0.5)
    
    if propulsion_detected:
        print("🎉 SUCCESS: Neural network discovered a PROPULSIVE warp drive!")
    else:
        print("⚠️  Partial success: Warp bubble exists but propulsion needs improvement")
    
    # Create smooth 3D visualization
    print("\n=== CREATING 3D VISUALIZATION ===")
    visualizer = SmoothWarpBubble3DVisualizer(model)
    animation = visualizer.create_smooth_3d_animation(duration_seconds=10, fps=15)
    
    # Save animation
    try:
        animation.save('propulsive_warp_drive.gif', writer='pillow', fps=15, dpi=100)
        print("Animation saved as 'propulsive_warp_drive.gif'")
    except Exception as e:
        print(f"Could not save animation: {e}")
    
    # Save model
    torch.save(model.state_dict(), 'propulsive_warp_drive_model.pth')
    print("Model saved as 'propulsive_warp_drive_model.pth'")
    
    print("\n=== SIMULATION COMPLETE ===")
    print("Displaying animation...")
    plt.show()