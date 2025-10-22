

class LEOWindowManager:
    def __init__(self, start_ts: float, T_max_sec: float, gap_sec: float = 120.0):
        self.T_max = float(max(0.0, T_max_sec))
        self.gap = float(max(0.0, gap_sec))
        self.period = self.T_max + self.gap
        self.start = float(start_ts)

    def is_available(self, ts: float) -> bool:
        if self.period <= 0.0 or self.T_max <= 0.0:
            return False
        phase = (ts - self.start) % self.period
        return phase < self.T_max

    def remaining_available(self, ts: float) -> float:
        if not self.is_available(ts):
            return 0.0
        phase = (ts - self.start) % self.period
        return max(0.0, self.T_max - phase)# time left
# =============================
# Satellite visibility utilities - CORRECTED VERSION
# =============================
import numpy as np
from dataclasses import dataclass

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

class SatelliteParams:
    def __init__(self, R_E_km: float = 6378.0, H_km: float = 1300.0, 
                 eps_min_deg: float = 10, inc_deg: float = 0.0,
                 omega_s_deg_s: float = None, omega_E_deg_s: float = 0.004178):
        self.R_E_km = R_E_km
        self.H_km = H_km
        self.eps_min_deg = eps_min_deg
        self.inc_deg = inc_deg
        
        # Calculate satellite angular velocity based on altitude
        if omega_s_deg_s is None:
            # ω = sqrt(GM / r^3) where r = R_E + H
            # For circular orbit: T = 2π * sqrt(r^3 / GM)
            # GM ≈ 398600 km³/s² for Earth
            r_km = R_E_km + H_km
            orbital_period_sec = 2 * np.pi * np.sqrt(r_km**3 / 398600.0)
            self.omega_s_deg_s = 360.0 / orbital_period_sec
        else:
            self.omega_s_deg_s = omega_s_deg_s
            
        self.omega_E_deg_s = omega_E_deg_s
        
        # For equatorial orbit, pole is at (90°, 0°)
        self.pole_lat_deg = 90.0 - inc_deg  # Corrected pole calculation
        self.pole_lon_deg = 0.0  # Reference longitude

    def _beta_max_rad(self) -> float:
        """Compute β_max in radians using the correct geometric relationship."""
        eps = self.eps_min_deg * DEG2RAD
        
        # From the paper's Equation (1): sin α = (R_E / (R_E + H)) * cos ε
        ratio = self.R_E_km / (self.R_E_km + self.H_km)
        x = ratio * np.cos(eps)
        x = np.clip(x, -1.0, 1.0)
        alpha = np.arcsin(x)  # α in rad
        
        # From Equation (2): ε + α + β = 90°
        beta_max = (np.pi/2.0) - eps - alpha
        
        return beta_max

    def _beta_min_rad(self, user_lat_deg: float, user_lon_deg: float) -> float:
        """Compute β_min - the minimum central angle to target."""
        phi_user = user_lat_deg * DEG2RAD
        phi_pole = self.pole_lat_deg * DEG2RAD
        dlam = (user_lon_deg - self.pole_lon_deg) * DEG2RAD

        rhs = (np.sin(phi_pole) * np.sin(phi_user) + 
               np.cos(phi_pole) * np.cos(phi_user) * np.cos(dlam))
        rhs = np.clip(rhs, -1.0, 1.0)
        return np.arcsin(rhs)

    def is_visible(self, user_lat_deg: float, user_lon_deg: float) -> bool:
        """Check if location is geometrically visible from satellite orbit."""
        beta_max = self._beta_max_rad()
        beta_min = self._beta_min_rad(user_lat_deg, user_lon_deg)
        
        # Location is visible if β_min ≤ β_max
        return beta_min <= beta_max

    def visibility_time_seconds(self, user_lat_deg: float, user_lon_deg: float) -> float:
        """Calculate pass duration for a SINGLE satellite."""
        if not self.is_visible(user_lat_deg, user_lon_deg):
            return 0.0
            
        beta_max = self._beta_max_rad()
        beta_min = self._beta_min_rad(user_lat_deg, user_lon_deg)
        
        # For equatorial orbit, we can use simplified calculation
        # The maximum latitude visible is approximately β_max
        max_visible_lat_deg = beta_max * RAD2DEG
        
        if abs(user_lat_deg) > max_visible_lat_deg:
            return 0.0
            
        # Simplified time calculation for equatorial orbit
        # Pass time ≈ (orbital_period / 180°) * arccos(cos(β_max) / cos(|lat|))
        cos_bmax = np.cos(beta_max)
        user_lat_rad = abs(user_lat_deg) * DEG2RAD
        cos_lat = np.cos(user_lat_rad)
        
        if cos_lat < 1e-8:
            return 0.0
            
        arg = cos_bmax / cos_lat
        arg = np.clip(arg, -1.0, 1.0)
        
        # Angular extent of pass
        pass_angle_rad = 2.0 * np.arccos(arg)
        
        # Convert to time
        orbital_period_sec = 360.0 / self.omega_s_deg_s
        T_pass_sec = (pass_angle_rad / (2 * np.pi)) * orbital_period_sec
        
        return float(max(0.0, T_pass_sec))