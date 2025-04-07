import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import openpyxl
from scipy.integrate import simpson

class ParticleMeshSimulation:
    def __init__(self, grid_size, num_particles):
        self.grid_size = grid_size
        self.num_particles = num_particles

        self.positions = np.zeros((self.num_particles, 3))
        self.velocities = np.zeros((self.num_particles, 3))

        self.density_grid = np.zeros((grid_size + 1, grid_size + 1, grid_size + 1))
        self.potential_grid = np.zeros_like(self.density_grid)
        self.force_grid = np.zeros((3, grid_size + 1, grid_size + 1, grid_size + 1))

        self.a = 1
        self.ns = 0.96

        self.omega_m = 1
        self.omega_k = 0
        self.omega_lambda = 0

        self.fb = 0.049
        self.sigma_8 = 0.8 * (self.omega_m / 0.3) ** 0.5

    def initial_positions_grid_center(self):
        q = np.linspace(0, 32, self.grid_size, endpoint=False) + 0.5
        qx, qy, qz = np.meshgrid(q, q, q, indexing='ij')
        self.positions = np.column_stack((qx.ravel(), qy.ravel(), qz.ravel()))
        self.positions %= self.grid_size

    def hubble(self, a=None):
        if a == None:
            a = self.a
        return 70 * self.E(a)

    def E(self, a):
        return (self.omega_m * self.a ** (-3/2) + self.omega_k * self.a ** -2 + self.omega_lambda) ** 0.5

    def growth_factor(self):
        return self.a

    def power_spectrum(self, ks):
        h = 0.7
        omh2 = self.omega_m * h**2

        k_eq = 0.0764 * omh2
        k_silk = 0.1 * omh2**(3/4)
        s = 44.5 * np.log(9.83/omh2) / np.sqrt(1 + 10*omh2**(3/4))
        
        q = ks / (self.omega_m * h * k_eq)
        Tm = (np.log(1 + 2.34*q)/(2.34*q) * (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-1/4))
        
        Tb = (np.sin(ks * s)/(ks*s)) * np.exp(-(ks / k_silk)**2)
        
        T = self.fb * Tb + (1 - self.fb) * Tm
        
        P = np.where(
            ks < k_eq,
            T * ks**self.ns,
            T * ks**(self.ns - 4) * np.log(ks/k_silk)
        )
        
        R = 8 / h
        kR = ks * R
        W = 3*(np.sin(kR) - kR*np.cos(kR))/kR**3
        
        integrand = P * W**2 * ks **2 / (2*np.pi**2)
        I = simpson(integrand, ks)
        
        A = self.sigma_8**2 / I
        return P * A

    def zeldovich(self, delta_a=0.01):
        k_freq = 2*np.pi*np.fft.fftfreq(self.grid_size + 1, d=1.0)
        kx, ky, kz = np.meshgrid(k_freq, k_freq, k_freq, indexing='ij')
        k2 = kx**2 + ky**2 + kz**2
        k2[0,0,0] = 1

        rand_field = np.random.normal(0, 1, (self.grid_size + 1, self.grid_size + 1, self.grid_size + 1)) + 1j * np.random.normal(0, 1, (self.grid_size + 1, self.grid_size + 1, self.grid_size + 1))
        rand_field = np.fft.fftshift(rand_field)
        rand_field[0,0,0] = 0
        
        k_mag = np.sqrt(k2)
        P_k = self.power_spectrum(k_mag.ravel()).reshape(k2.shape)
        delta_k = np.sqrt(P_k) * rand_field
        
        phi_k = delta_k * (-1/k2)
        phi_k[0,0,0] = 0
        self.potential_grid = np.fft.ifftn(phi_k).real
        
        self.compute_forces()
        forces = self.interpolate_forces('CIC')
        
        D = self.growth_factor()
        f = D * self.a * self.hubble()
        self.positions += forces * D
        self.velocities = -forces * f * D
        
        self.positions %= self.grid_size

        print('Zel\'dovich approximation completed.')

    def mass_assign(self, scheme):
        self.density_grid = np.zeros((self.grid_size + 1, self.grid_size + 1, self.grid_size + 1))
        if scheme == 'CIC':
            for pos in self.positions:
                i, j, k = np.floor(pos).astype(int) % self.grid_size

                dx, dy, dz = pos - np.array([i, j, k])
                ip1, jp1, kp1 = (i + 1) % self.grid_size, (j + 1) % self.grid_size, (k + 1) % self.grid_size
                w000 = (1 - dx) * (1 - dy) * (1 - dz)
                w100 = dx * (1 - dy) * (1 - dz)
                w010 = (1 - dx) * dy * (1 - dz)
                w001 = (1 - dx) * (1 - dy) * dz
                w110 = dx * dy * (1 - dz)
                w101 = dx * (1 - dy) * dz
                w011 = (1 - dx) * dy * dz
                w111 = dx * dy * dz

                self.density_grid[i, j, k] += w000
                self.density_grid[ip1, j, k] += w100
                self.density_grid[i, jp1, k] += w010
                self.density_grid[i, j, kp1] += w001
                self.density_grid[ip1, jp1, k] += w110
                self.density_grid[ip1, j, kp1] += w101
                self.density_grid[i, jp1, kp1] += w011
                self.density_grid[ip1, jp1, kp1] += w111

        if scheme == 'NGP':  
            for pos in self.positions:
                i, j, k = np.floor(pos + 0.5).astype(int)
                i %= self.grid_size
                j %= self.grid_size
                k %= self.grid_size
                self.density_grid[i, j, k] += 1

        if scheme == 'TSC':
            for pos in self.positions:
                i, j, k = np.floor(pos).astype(int) % self.grid_size

                dx, dy, dz = pos - np.array([i, j, k])

                im1, jm1, km1 = (i - 1) % self.grid_size, (j - 1) % self.grid_size, (k - 1) % self.grid_size
                ip1, jp1, kp1 = (i + 1) % self.grid_size, (j + 1) % self.grid_size, (k + 1) % self.grid_size

                # Compute weights
                wx = np.array([0.5 * (1 - dx) ** 2, 0.75 - (dx - 0.5) ** 2, 0.5 * dx ** 2])
                wy = np.array([0.5 * (1 - dy) ** 2, 0.75 - (dy - 0.5) ** 2, 0.5 * dy ** 2])
                wz = np.array([0.5 * (1 - dz) ** 2, 0.75 - (dz - 0.5) ** 2, 0.5 * dz ** 2])

                grid_indices_x = np.array([im1, i, ip1])
                grid_indices_y = np.array([jm1, j, jp1])
                grid_indices_z = np.array([km1, k, kp1])

                for xi, wxi in zip(grid_indices_x, wx):
                    for yi, wyi in zip(grid_indices_y, wy):
                        for zi, wzi in zip(grid_indices_z, wz):
                            self.density_grid[xi, yi, zi] += wxi * wyi * wzi

        self.density_grid -= 1

    def solve_poisson(self):
        rho_k = np.fft.fftn(self.density_grid)
        kx, ky, kz = np.meshgrid(
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            indexing='ij'
        )

        k2 = kx ** 2 + ky ** 2 + kz ** 2
        k2[0, 0, 0] = 1
        greens_function = -1 / k2
        greens_function[0, 0, 0] = 0

        phi_k = rho_k * greens_function
        self.potential_grid = np.fft.ifftn(phi_k).real

    def solve_poisson_2(self):
        rho_k = fftn(self.density_grid)
        kx, ky, kz = np.meshgrid(
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            indexing='ij'
        )
        sin2_kx = np.sin(kx * self.grid_spacing / 2) ** 2
        sin2_ky = np.sin(ky * self.grid_spacing / 2) ** 2 
        sin2_kz = np.sin(kz * self.grid_spacing / 2) ** 2
        denominator = sin2_kx + sin2_ky + sin2_kz
        denominator[0, 0, 0] = 1

        greens_function = -self.grid_spacing ** 2 / ( 4 * denominator)
        greens_function[0, 0, 0] = 0

        phi_k = rho_k * greens_function
        self.potential_grid = np.real(ifftn(phi_k))

    def compute_forces(self):
        self.force_grid[0, 1:-1, :, :] = -(self.potential_grid[2:, :, :] - self.potential_grid[:-2, :, :]) / 2
        self.force_grid[1, :, 1:-1, :] = -(self.potential_grid[:, 2:, :] - self.potential_grid[:, :-2, :]) / 2
        self.force_grid[2, :, :, 1:-1] = -(self.potential_grid[:, :, 2:] - self.potential_grid[:, :, :-2]) / 2

        self.force_grid[0, 0, :, :] = -(self.potential_grid[1, :, :] - self.potential_grid[-1, :, :]) / 2
        self.force_grid[0, -1, :, :] = -(self.potential_grid[0, :, :] - self.potential_grid[-2, :, :]) / 2
 
        self.force_grid[1, :, 0, :] = -(self.potential_grid[:, 1, :] - self.potential_grid[:, -1, :]) / 2
        self.force_grid[1, :, -1, :] = -(self.potential_grid[:, 0, :] - self.potential_grid[:, -2, :]) / 2

        self.force_grid[2, :, :, 0] = -(self.potential_grid[:, :, 1] - self.potential_grid[:, :, -1]) / 2
        self.force_grid[2, :, :, -1] = -(self.potential_grid[:, :, 0] - self.potential_grid[:, :, -2]) / 2

    def interpolate_forces(self, scheme):
        forces = np.zeros((self.num_particles, 3))

        if scheme == 'NGP':
            for n in range(self.num_particles):
                pos = self.positions[n]
                i, j, k = np.floor(pos + 0.5).astype(int)
                i %= self.grid_size
                j %= self.grid_size
                k %= self.grid_size
                for axis in range(3):
                    forces[n, axis] = self.force_grid[axis, i, j, k]

        if scheme == 'CIC':
            for n in range(self.num_particles):
                pos = self.positions[n]
                i, j, k = np.floor(pos).astype(int)

                dx, dy, dz = pos - np.array([i, j, k])
                ip1, jp1, kp1 = i + 1, j + 1, k + 1

                w000 = (1 - dx) * (1 - dy) * (1 - dz)
                w100 = dx * (1 - dy) * (1 - dz)
                w010 = (1 - dx) * dy * (1 - dz)
                w001 = (1 - dx) * (1 - dy) * dz
                w110 = dx * dy * (1 - dz)
                w101 = dx * (1 - dy) * dz
                w011 = (1 - dx) * dy * dz
                w111 = dx * dy * dz
                for axis in range(3):
                    forces[n, axis] = (
                        w000 * self.force_grid[axis, i, j, k]+
                        w100 * self.force_grid[axis, ip1, j, k]+
                        w010 * self.force_grid[axis, i, jp1, k]+
                        w001 * self.force_grid[axis, i, j, kp1]+
                        w110 * self.force_grid[axis, ip1, jp1, k]+
                        w101 * self.force_grid[axis, ip1, j, kp1]+
                        w011 * self.force_grid[axis, i, jp1, kp1]+
                        w111 * self.force_grid[axis, ip1, jp1, kp1]
                    )

        if scheme == 'TSC':
            for axis in range(3):
                for pos in self.positions:
                    ix0 = np.floor(pos[0]).astype(int)
                    iy0 = np.floor(pos[1]).astype(int)
                    iz0 = np.floor(pos[2]).astype(int)

                    dx = pos[0] - ix0
                    dy = pos[1] - iy0
                    dz = pos[2] - iz0

                    # Optimized TSC weight functions
                    wx = np.array([0.5 * (1 - dx) ** 2, 0.75 - (dx - 0.5) ** 2, 0.5 * dx ** 2])
                    wy = np.array([0.5 * (1 - dy) ** 2, 0.75 - (dy - 0.5) ** 2, 0.5 * dy ** 2])
                    wz = np.array([0.5 * (1 - dz) ** 2, 0.75 - (dz - 0.5) ** 2, 0.5 * dz ** 2])

                    # Compute neighboring indices safely
                    grid_indices_x = np.clip(np.array([ix0 - 1, ix0, ix0 + 1]), 0, self.grid_size - 1)
                    grid_indices_y = np.clip(np.array([iy0 - 1, iy0, iy0 + 1]), 0, self.grid_size - 1)
                    grid_indices_z = np.clip(np.array([iz0 - 1, iz0, iz0 + 1]), 0, self.grid_size - 1)

                    # Compute interpolated force
                    for xi, wxi in zip(grid_indices_x, wx):
                        for yi, wyi in zip(grid_indices_y, wy):
                            for zi, wzi in zip(grid_indices_z, wz):
                                forces[:, axis] += wxi * wyi * wzi * self.force_grid[axis, xi, yi, zi]

        return forces

    def leapfrog(self, delta_a, scheme):
        forces = self.interpolate_forces(scheme)
        self.velocities += (3 * forces * delta_a / (4 * (self.a + delta_a * 0.5) ** 2 * self.E(self.a + delta_a * 0.5)))
        self.positions += self.velocities * delta_a / (self.E(self.a) * (self.a) ** 3)
        self.positions %= self.grid_size
        self.a += delta_a

    def kinetic(self):
        return 0.5 * np.sum(self.velocities**2)

    def potential(self):
        return 0.5 * np.sum(self.density_grid * self.potential_grid)

    def visualize(self, title_extra=""):
        plt.figure(figsize=(8, 8))
        
        plt.scatter(
            self.positions[:, 0], 
            self.positions[:, 1], 
            s=0.5,
            alpha=0.6,
        )
        plt.xlabel('X Position', fontsize=12)
        plt.ylabel('Y Position', fontsize=12)
        plt.show()

    def visualize_potential(self, slice_index=None, projection='2d'):
        if slice_index is None:
            slice_index = self.grid_size // 2

        plt.figure(figsize=(8, 6))
        plt.imshow(self.potential_grid[slice_index, :, :], cmap='viridis', origin='lower',
                   extent=[0, 1, 0, 1], aspect='auto')
        plt.colorbar(label='Potential')
        plt.title(f'Potential Field (Slice at index {slice_index})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()


    def compute_power_spectrum(self, delta_x, n_bins=20):
        N = delta_x.shape[0]
        delta_k = np.fft.fftn(delta_x)
        kx = np.fft.fftfreq(N, d=self.grid_size/N) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=self.grid_size/N) * 2 * np.pi
        kz = np.fft.fftfreq(N, d=self.grid_size/N) * 2 * np.pi
        k_grid = np.sqrt(kx[:, None, None]**2 + ky[None, :, None]**2 + kz[None, None, :]**2)
        
        k_max = np.max(k_grid)
        k_bins = np.logspace(np.log10(2 * np.pi / self.grid_size), np.log10(k_max), n_bins)
        power = np.histogram(k_grid, bins=k_bins, weights=np.abs(delta_k)**2)[0]
        counts = np.histogram(k_grid, bins=k_bins)[0]

        mask = counts > 0
        k_centers = 0.5 * (k_bins[1:] + k_bins[:-1])
        P_k = (self.grid_size ** 3 / N ** 6) * (power[mask] / counts[mask])
        
        return k_centers[mask], P_k


    def solve_telegraph(self, da, kappa=2.5, c_g=1.0, max_iter=100, tolerance=1e-4):
        psi = np.zeros_like(self.potential_grid)
        H = self.hubble()
        dHda = -1 * (self.hubble(a=self.a) - self.hubble(a=self.a + da)) / da
        laplacian_coeff = 1 / (self.a * H) ** 2
        damping_coeff = (1/self.a + dHda / H + 2 * kappa / (self.a * H))
        for iteration in range(max_iter):
            # Update Φ: ∂Φ/∂a = Ψ
            self.potential_grid += da * psi
            laplacian = np.zeros_like(self.potential_grid)
            phi_xp = np.roll(self.potential_grid, -1, axis=0)
            phi_xm = np.roll(self.potential_grid, 1, axis=0)
            phi_yp = np.roll(self.potential_grid, -1, axis=1)
            phi_ym = np.roll(self.potential_grid, 1, axis=1)
            phi_zp = np.roll(self.potential_grid, -1, axis=2)
            phi_zm = np.roll(self.potential_grid, 1, axis=2)

            laplacian = (phi_xp + phi_xm + phi_yp + phi_ym + phi_zp + phi_zm - 6 * self.potential_grid)


            # Update Ψ: ∂Ψ/∂a = -damping_coeff*Ψ + laplacian_coeff*∇²Φ - source_coeff
            psi += da * (-damping_coeff * psi + laplacian_coeff * laplacian - laplacian_coeff * self.density_grid)

        return iteration + 1

    def _compute_laplacian(self, field):
        field_k = np.fft.fftn(field)
        kx, ky, kz = np.meshgrid(
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            np.fft.fftfreq(self.grid_size + 1) * 2 * np.pi,
            indexing='ij'
        )
        k2 = kx**2 + ky**2 + kz**2
        k2[0, 0, 0] = 1
        laplacian_k = -k2 * field_k
        
        return np.fft.ifftn(laplacian_k).real

   