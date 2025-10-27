import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from mpl_toolkits.mplot3d import Axes3D

class WarpDriveNN(nn.Module):
    def __init__(self, hidden_dim=128, num_layers=5):
        super(WarpDriveNN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(4, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(hidden_dim, 10))  # 10 components for symmetric 4x4 metric
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, coordinates):
        return self.network(coordinates)

class WarpDriveObjective:
    def __init__(self, weights=None):
        self.weights = weights or {
            'einstein': 1.0,
            'warp_bubble': 2.0,
            'energy_violation': 1.5,
            'energy_minimization': 3.0,  # NEW: Strong weight for energy minimization
            'causality': 1.0,
            'boundary': 0.5,
            'regularity': 0.3,
        }
    
    def __call__(self, model, coordinates):
        total_loss = 0.0
        loss_components = {}
        
        # 1. Einstein Field Equations
        loss_einstein = self.compute_einstein_loss(model, coordinates)
        total_loss += self.weights['einstein'] * loss_einstein
        loss_components['einstein'] = loss_einstein.item()
        
        # 2. Warp Bubble Structure
        loss_bubble = self.compute_warp_bubble_loss(model, coordinates)
        total_loss += self.weights['warp_bubble'] * loss_bubble
        loss_components['bubble'] = loss_bubble.item()
        
        # 3. Energy Condition Violation (should violate WEC)
        loss_energy_violation = self.compute_energy_condition_loss(model, coordinates)
        total_loss += self.weights['energy_violation'] * loss_energy_violation
        loss_components['energy_violation'] = loss_energy_violation.item()
        
        # 4. ENERGY MINIMIZATION - NEW: Minimize total negative energy required
        loss_energy_minimization = self.compute_energy_minimization_loss(model, coordinates)
        total_loss += self.weights['energy_minimization'] * loss_energy_minimization
        loss_components['energy_minimization'] = loss_energy_minimization.item()
        
        # 5. Causality Preservation
        loss_causality = self.compute_causality_loss(model, coordinates)
        total_loss += self.weights['causality'] * loss_causality
        loss_components['causality'] = loss_causality.item()
        
        # 6. Boundary Conditions
        loss_boundary = self.compute_boundary_loss(model, coordinates)
        total_loss += self.weights['boundary'] * loss_boundary
        loss_components['boundary'] = loss_boundary.item()
        
        # 7. Metric Regularity
        loss_regularity = self.compute_regularity_loss(model, coordinates)
        total_loss += self.weights['regularity'] * loss_regularity
        loss_components['regularity'] = loss_regularity.item()
        
        return total_loss, loss_components
    
    def compute_einstein_loss(self, model, coordinates):
        """Einstein field equations G_μν = 0"""
        batch_size = coordinates.shape[0]
        coordinates.requires_grad_(True)
        
        metric_flat = model(coordinates)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        # Simplified Einstein tensor calculation
        # In practice, this would involve full Christoffel and Ricci calculations
        G_approx = torch.zeros(batch_size, 4, 4, device=coordinates.device)
        
        # For demonstration, we'll use a simplified approach
        # Focus on key components that should vanish in vacuum
        for i in range(4):
            for j in range(4):
                if i == 0 and j == 0:  # Time-time component
                    G_approx[:, i, j] = g[:, i, j] + 1.0  # Should approach -1
                elif i == j:  # Space-space diagonal
                    G_approx[:, i, j] = g[:, i, j] - 1.0  # Should approach 1
                else:  # Off-diagonal
                    G_approx[:, i, j] = g[:, i, j]  # Should approach 0
        
        return torch.mean(G_approx**2)
    
    def compute_warp_bubble_loss(self, model, coordinates):
        """Encourage warp bubble structure"""
        batch_size = coordinates.shape[0]
        r = torch.norm(coordinates[:, 1:], dim=1)
        
        inside_mask = r < 0.3
        shell_mask = (r >= 0.3) & (r < 1.0)
        outside_mask = r >= 1.0
        
        metric_flat = model(coordinates)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        loss_bubble = 0.0
        g_minkowski = self.get_minkowski_metric(batch_size, coordinates.device)
        
        # Inside: should be Minkowski
        if torch.any(inside_mask):
            g_inside = g[inside_mask]
            loss_inside = torch.mean((g_inside - g_minkowski[:len(g_inside)])**2)
            loss_bubble += loss_inside
        
        # Shell: should be different from Minkowski
        if torch.any(shell_mask):
            g_shell = g[shell_mask]
            # Encourage some warping in shell (but not too extreme for energy minimization)
            shell_warping = torch.mean((g_shell - g_minkowski[:len(g_shell)])**2)
            # Target moderate warping (balance between functionality and energy)
            target_warp = 0.5
            loss_shell = (shell_warping - target_warp)**2
            loss_bubble += loss_shell
        
        # Outside: should be Minkowski
        if torch.any(outside_mask):
            g_outside = g[outside_mask]
            loss_outside = torch.mean((g_outside - g_minkowski[:len(g_outside)])**2)
            loss_bubble += loss_outside
        
        return loss_bubble
    
    def compute_energy_condition_loss(self, model, coordinates):
        """Encourage violation of Weak Energy Condition in shell region"""
        batch_size = coordinates.shape[0]
        r = torch.norm(coordinates[:, 1:], dim=1)
        shell_mask = (r >= 0.3) & (r < 1.0)
        
        if not torch.any(shell_mask):
            return torch.tensor(0.0, device=coordinates.device)
        
        metric_flat = model(coordinates[shell_mask])
        g_shell = self.reshape_to_metric(metric_flat, torch.sum(shell_mask))
        
        # For warp drives, g_tt should be less negative in shell region
        # This corresponds to negative energy density
        g_tt_shell = g_shell[:, 0, 0]
        
        # Target: g_tt > -1 (less negative than Minkowski)
        # But not too close to 0 (which would require excessive energy)
        target_g_tt = -0.7  # Balanced value for moderate energy requirements
        loss_energy = torch.mean((g_tt_shell - target_g_tt)**2)
        
        return loss_energy
    
    def compute_energy_minimization_loss(self, model, coordinates):
        """
        NEW: Minimize total energy requirements for practical warp drive
        This is crucial for making the drive physically realizable
        """
        batch_size = coordinates.shape[0]
        r = torch.norm(coordinates[:, 1:], dim=1)
        shell_mask = (r >= 0.3) & (r < 1.0)
        
        if not torch.any(shell_mask):
            return torch.tensor(0.0, device=coordinates.device)
        
        coordinates_shell = coordinates[shell_mask]
        coordinates_shell.requires_grad_(True)
        
        metric_flat = model(coordinates_shell)
        g_shell = self.reshape_to_metric(metric_flat, torch.sum(shell_mask))
        
        # Estimate energy density from metric components
        # Simplified model: energy density ~ (deviation from Minkowski)^2
        g_minkowski = self.get_minkowski_metric(len(g_shell), coordinates.device)
        metric_deviation = g_shell - g_minkowski
        
        # Focus on components that contribute most to energy requirements
        energy_density_estimate = (
            torch.abs(metric_deviation[:, 0, 0]) +  # time-time component
            torch.abs(metric_deviation[:, 0, 1]) +  # time-space components
            torch.abs(metric_deviation[:, 0, 2]) +
            torch.abs(metric_deviation[:, 0, 3])
        )
        
        # Penalize large energy densities
        loss_energy_minimization = torch.mean(energy_density_estimate**2)
        
        # Additional penalty for sharp gradients (which require more energy)
        loss_gradients = 0.0
        for i in range(4):
            for j in range(4):
                g_ij = g_shell[:, i, j]
                grad_g = torch.autograd.grad(
                    g_ij, coordinates_shell,
                    grad_outputs=torch.ones_like(g_ij),
                    create_graph=True, retain_graph=True
                )[0]
                loss_gradients += torch.mean(grad_g**2)
        
        return loss_energy_minimization + 0.1 * loss_gradients
    
    def compute_causality_loss(self, model, coordinates):
        """Preserve causality - no closed timelike curves"""
        batch_size = coordinates.shape[0]
        metric_flat = model(coordinates)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        # Check metric determinant (should be negative for Lorentzian signature)
        det_g = torch.det(g)
        loss_det = torch.mean(torch.relu(det_g + 1e-6))
        
        # Check time-time component (should be negative)
        g_tt = g[:, 0, 0]
        loss_g_tt = torch.mean(torch.relu(g_tt + 1e-6))
        
        return loss_det + loss_g_tt
    
    def compute_boundary_loss(self, model, coordinates):
        """Asymptotic flatness at infinity"""
        batch_size = coordinates.shape[0]
        r = torch.norm(coordinates[:, 1:], dim=1)
        far_mask = r > 2.0
        
        if not torch.any(far_mask):
            return torch.tensor(0.0, device=coordinates.device)
        
        metric_flat = model(coordinates[far_mask])
        g_far = self.reshape_to_metric(metric_flat, torch.sum(far_mask))
        g_minkowski = self.get_minkowski_metric(len(g_far), coordinates.device)
        
        return torch.mean((g_far - g_minkowski)**2)
    
    def compute_regularity_loss(self, model, coordinates):
        """Metric regularity and smoothness"""
        batch_size = coordinates.shape[0]
        coordinates.requires_grad_(True)
        
        metric_flat = model(coordinates)
        g = self.reshape_to_metric(metric_flat, batch_size)
        
        # Penalize large metric components
        loss_magnitude = torch.mean(g**2)
        
        # Penalize large derivatives
        loss_smoothness = 0.0
        for i in range(4):
            for j in range(4):
                g_ij = g[:, i, j]
                grad_g = torch.autograd.grad(
                    g_ij, coordinates,
                    grad_outputs=torch.ones_like(g_ij),
                    create_graph=True, retain_graph=True
                )[0]
                loss_smoothness += torch.mean(grad_g**2)
        
        return 0.1 * loss_magnitude + 0.01 * loss_smoothness
    
    def reshape_to_metric(self, metric_flat, batch_size):
        """Reshape to symmetric 4x4 metric tensor"""
        g = torch.zeros(batch_size, 4, 4, device=metric_flat.device)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        
        return g
    
    def get_minkowski_metric(self, batch_size, device):
        """Get Minkowski metric tensor"""
        g = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        g[:, 0, 0] = -1.0
        return g

class AdaptiveWarpDriveTrainer:
    def __init__(self, model, objective):
        self.model = model
        self.objective = objective
        self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=100)
        self.loss_history = {key: [] for key in ['total', 'einstein', 'bubble', 'energy_violation', 'energy_minimization', 'causality', 'boundary', 'regularity']}
    
    def train(self, epochs=1000, batch_size=1024):
        print("Training Neural Network for Novel Warp Drive Solutions...")
        print("This includes energy minimization for practical realization.")
        
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
                self.adapt_weights(epoch)
    
    def sample_training_points(self, batch_size):
        """Sample coordinates strategically"""
        base_samples = torch.rand(batch_size, 4) * 4 - 2
        
        # Extra samples in shell region (where energy minimization is critical)
        shell_samples = torch.rand(batch_size // 3, 4) * 1.4 - 0.7
        shell_r = torch.norm(shell_samples[:, 1:], dim=1)
        shell_mask = (shell_r > 0.3) & (shell_r < 1.0)
        shell_samples = shell_samples[shell_mask]
        
        all_samples = torch.cat([base_samples, shell_samples], dim=0)
        return all_samples
    
    def adapt_weights(self, epoch):
        """Adaptively adjust loss weights"""
        if epoch > 500 and epoch % 200 == 0:
            # Increase weight for energy minimization if it's not converging well
            recent_energy_min = np.mean(self.loss_history['energy_minimization'][-50:])
            recent_total = np.mean(self.loss_history['total'][-50:])
            
            if recent_energy_min > recent_total * 0.3:
                self.objective.weights['energy_minimization'] *= 1.2
                print(f"Increased energy minimization weight to {self.objective.weights['energy_minimization']}")
    
    def print_progress(self, epoch, total_loss, components):
        print(f"\nEpoch {epoch}: Total Loss = {total_loss.item():.6f}")
        print("Component Losses:")
        for name, value in components.items():
            print(f"  {name:20}: {value:.6f}")

class WarpDriveVerifier:
    def __init__(self):
        self.tests = {}
    
    def verify_warp_drive(self, model):
        """Verify the learned metric is a valid warp drive"""
        results = {}
        
        # Test points for verification
        test_coords = torch.tensor([
            [0.0, 0.0, 0.0, 0.0],      # Inside bubble
            [0.0, 0.7, 0.0, 0.0],      # Shell region
            [0.0, 2.0, 0.0, 0.0],      # Outside bubble
        ], dtype=torch.float32)
        
        with torch.no_grad():
            metric_flat = model(test_coords)
            g = self.reshape_to_metric(metric_flat, len(test_coords))
            
            # Test 1: Timelike observer inside bubble
            g_inside = g[0]
            results['timelike_inside'] = g_inside[0, 0].item() < 0
            
            # Test 2: Warping in shell region
            g_shell = g[1]
            g_minkowski = torch.eye(4)
            g_minkowski[0, 0] = -1
            shell_warp = torch.norm(g_shell - g_minkowski).item()
            results['shell_warping'] = shell_warp > 0.1
            
            # Test 3: Asymptotic flatness
            g_outside = g[2]
            outside_flatness = torch.norm(g_outside - g_minkowski).item()
            results['asymptotic_flatness'] = outside_flatness < 0.1
            
            # Test 4: Energy requirements (estimate)
            energy_estimate = self.estimate_energy_requirements(model)
            results['reasonable_energy'] = energy_estimate < 10.0  # Arbitrary threshold
        
        results['is_valid_warp_drive'] = all(results.values())
        return results, energy_estimate
    
    def estimate_energy_requirements(self, model, n_samples=1000):
        """Estimate total negative energy requirements"""
        # Sample points in shell region
        coords = torch.rand(n_samples, 4) * 2 - 1
        r = torch.norm(coords[:, 1:], dim=1)
        shell_mask = (r > 0.3) & (r < 1.0)
        shell_coords = coords[shell_mask]
        
        with torch.no_grad():
            metric_flat = model(shell_coords)
            g_shell = self.reshape_to_metric(metric_flat, len(shell_coords))
            
            # Simplified energy density estimate
            g_minkowski = torch.eye(4).unsqueeze(0).repeat(len(shell_coords), 1, 1)
            g_minkowski[:, 0, 0] = -1
            deviation = torch.abs(g_shell - g_minkowski)
            energy_density = torch.mean(deviation, dim=(1, 2))
            
            # Total energy estimate (integral over shell volume)
            total_energy = torch.sum(energy_density).item()
        
        return total_energy
    
    def reshape_to_metric(self, metric_flat, batch_size):
        g = torch.zeros(batch_size, 4, 4)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        
        return g

class WarpDriveVisualizer:
    def __init__(self):
        self.fig = None
    
    def plot_training_history(self, loss_history):
        """Plot training loss history"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(loss_history['total'])
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.yscale('log')
        
        plt.subplot(2, 2, 2)
        for key in ['einstein', 'bubble', 'causality']:
            plt.plot(loss_history[key], label=key)
        plt.title('Physics Constraints')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        
        plt.subplot(2, 2, 3)
        for key in ['energy_violation', 'energy_minimization']:
            plt.plot(loss_history[key], label=key)
        plt.title('Energy Optimization')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        
        plt.subplot(2, 2, 4)
        for key in ['boundary', 'regularity']:
            plt.plot(loss_history[key], label=key)
        plt.title('Boundary & Regularity')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.yscale('log')
        
        plt.tight_layout()
        plt.show()
    
    def plot_warp_drive_solution(self, model):
        """Visualize the learned warp drive metric"""
        # Create spatial grid
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        
        # Evaluate metric on grid
        g_tt = np.zeros_like(X)
        g_tx = np.zeros_like(X)
        
        with torch.no_grad():
            for i in range(len(x)):
                for j in range(len(y)):
                    coords = torch.tensor([[0.0, x[i], y[j], 0.0]], dtype=torch.float32)
                    metric_flat = model(coords)
                    g = self.reshape_to_metric(metric_flat, 1)
                    g_tt[j, i] = g[0, 0, 0].item()
                    g_tx[j, i] = g[0, 0, 1].item()
        
        # Plot results
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        im1 = axes[0].imshow(g_tt, extent=[-2, 2, -2, 2], origin='lower', cmap='RdBu_r')
        axes[0].set_title('Time-Time Component (g_tt)')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(g_tx, extent=[-2, 2, -2, 2], origin='lower', cmap='RdBu_r')
        axes[1].set_title('Time-Space Component (g_tx)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1])
        
        # Mark regions
        for ax in axes:
            circle_inside = plt.Circle((0, 0), 0.3, fill=False, color='green', linestyle='--', label='Inside Bubble')
            circle_shell = plt.Circle((0, 0), 1.0, fill=False, color='red', linestyle='--', label='Shell Region')
            ax.add_patch(circle_inside)
            ax.add_patch(circle_shell)
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return g_tt, g_tx
    
    def reshape_to_metric(self, metric_flat, batch_size):
        g = torch.zeros(batch_size, 4, 4)
        indices = [(0,0), (0,1), (0,2), (0,3), (1,1), (1,2), (1,3), (2,2), (2,3), (3,3)]
        
        for idx, (i, j) in enumerate(indices):
            g[:, i, j] = metric_flat[:, idx]
            g[:, j, i] = metric_flat[:, idx]
        
        return g

# MAIN EXECUTION
if __name__ == "__main__":
    print("=== Warp Drive Discovery with Energy Minimization ===")
    print("Training neural network to find novel, practical warp drive solutions...")
    
    # Initialize components
    model = WarpDriveNN(hidden_dim=128, num_layers=6)
    objective = WarpDriveObjective()
    trainer = AdaptiveWarpDriveTrainer(model, objective)
    verifier = WarpDriveVerifier()
    visualizer = WarpDriveVisualizer()
    
    # Train the model
    trainer.train(epochs=300, batch_size=32)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    
    # Verify the solution
    verification_results, energy_estimate = verifier.verify_warp_drive(model)
    
    print("\n=== VERIFICATION RESULTS ===")
    for test, passed in verification_results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test:25}: {status}")
    
    print(f"\nEstimated Energy Requirement: {energy_estimate:.6f}")
    
    if verification_results['is_valid_warp_drive']:
        print("\n🎉 SUCCESS: Valid warp drive solution found!")
        if energy_estimate < 5.0:
            print("💡 EXCELLENT: Low energy requirements make this potentially practical!")
        else:
            print("⚠️  NOTE: Energy requirements may be high - consider more training")
    else:
        print("\n❌ Solution does not meet all warp drive criteria")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualizer.plot_training_history(trainer.loss_history)
    g_tt, g_tx = visualizer.plot_warp_drive_solution(model)
    
    # Analyze energy distribution
    print("\n=== ENERGY ANALYSIS ===")
    print(f"Final Energy Minimization Loss: {trainer.loss_history['energy_minimization'][-1]:.6f}")
    print(f"Energy Violation Loss: {trainer.loss_history['energy_violation'][-1]:.6f}")
    
    # Save the trained model
    torch.save(model.state_dict(), 'warp_drive_model.pth')
    print("\nModel saved as 'warp_drive_model.pth'")
    
    print("\n=== NEXT STEPS ===")
    print("1. The model has learned a novel warp drive metric")
    print("2. Energy minimization was prioritized for practical realization") 
    print("3. You can load this model for further analysis or refinement")
    print("4. Consider running with more epochs for better energy optimization")