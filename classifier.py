import numpy as np
from typing import List, Tuple, Literal, Dict, Optional
import random

class CoverageClassifier:
    def __init__(self, mode: str = "real_simulation", group_config: Optional[Dict] = None,rng: Optional[random.Random]=None):
        """
        Initialize classifier with different modes.
        
        Args:
            mode: "real_simulation" or "group_assignment"
            group_config: For group_assignment mode, specify group distribution
                         e.g., {'Group A': 0.5, 'Group B': 0.5} or {'Group A': 1.0}
        """
        self.mode = mode
        self.group_config = group_config
        self.rng = rng if rng is not None else np.random.default_rng(42)
        # RAT coverage boxes coordinates
        self.group_definitions = {
            'Group A': ['RAT-1', 'RAT-4', 'RAT-5'],
            'Group B': ['RAT-1', 'RAT-4', 'RAT-2'], 
            'Group D': ['RAT-1', 'RAT-3','RAT-4'],
            'Group C': ['RAT-1', 'RAT-4'],
        }
        self.rat_coverage = {
            'RAT-1': {'lat_min': 24.0, 'lat_max': 25.0, 'lon_min': 25.0, 'lon_max': 26},
            'RAT-4': {'lat_min': 24.0, 'lat_max': 25.0, 'lon_min': 25.0, 'lon_max': 26},
            'RAT-3': {'lat_min': 24.0, 'lat_max': 25.0, 'lon_min': 25.7, 'lon_max': 26},
            'RAT-5': {'lat_min': 24.0, 'lat_max': 25.0, 'lon_min': 25, 'lon_max': 25.25},
            'RAT-2': {'lat_min': 24.0, 'lat_max': 25.0, 'lon_min': 25.3, 'lon_max': 25.6}
        }
        
        # Group definitions from table
        
        
        
        self.lat_bounds=(24.0,25.0)
        # Longitude bounds
        self.lon_bounds = (25.0, 26.0)
        




        self._current_group_index = 0
        
        self._group_sequence = []
        self.group_definitions = {
            'Group A': ['RAT-1', 'RAT-4', 'RAT-5'],
            'Group B': ['RAT-1', 'RAT-4', 'RAT-2'], 
            'Group D': ['RAT-1', 'RAT-3'],
            'Group C': ['RAT-1', 'RAT-4'],
        }
        if self.mode == "group_assignment" and self.group_config:
            self._setup_group_sequence()

    def _setup_group_sequence(self):
        """Setup group sequence for round-robin assignment"""
        self._group_sequence = []
        for group, percentage in self.group_config.items():
            count = int(percentage * 100)  
            self._group_sequence.extend([group] * count)
        
        # Shuffle to avoid patterns
        self.rng.shuffle(self._group_sequence)
        self._current_group_index = 0

    def get_next_group(self):
        """Get next group in sequence for group assignment mode"""
        if not self._group_sequence:
            return 'Group A'  # Fallback
        
        group = self._group_sequence[self._current_group_index]
        self._current_group_index = (self._current_group_index + 1) % len(self._group_sequence)
        return group

    def generate_coordinates(self) -> Tuple[float, float]:
        """Generate coordinates based on current mode"""
        if self.mode == "real_simulation":
            return self._generate_prioritized_coordinates()
        elif self.mode == "group_assignment":
            if self.group_config:
                group = self.get_next_group()
                return self._generate_coordinates_for_group(group)
            else:
                # If no group config, fall back to real simulation
                return self._generate_prioritized_coordinates()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def get_user_coverage(self, lat: float, lon: float) -> Dict:
        """Get coverage information for a single user based on current mode"""
        available_rats = self._get_rats_by_coverage(lat, lon)
        group = self._get_group_by_rats(available_rats)
        
        return {
            'lat': lat,
            'lon': lon, 
            'available_rats': available_rats,
            'group': group
        }

    def _generate_prioritized_coordinates(self) -> Tuple[float, float]:
        """Generate coordinates using gamma distribution for latitude prioritization"""
        
        # Gamma distribution parameters for right-skewed 
        shape = 0.7    
        scale = 2.5
        
        # Generate gamma value and map to latitude range 
        gamma_val = self.rng.gamma(shape, scale)
        
        # Ensure within bounds
        lat = self.rng.uniform(*self.lat_bounds)
        lon = self.rng.uniform(*self.lon_bounds)
        
        return lat, lon

    def _get_rats_by_coverage(self, lat: float, lon: float) -> List[str]:
        """Check which RATs cover the given coordinates"""
        available_rats = []
        
        for rat, coverage in self.rat_coverage.items():
            if (coverage['lat_min'] <= lat <= coverage['lat_max'] and
                coverage['lon_min'] <= lon <= coverage['lon_max']):
                available_rats.append(rat)
                
        # RAT-1 is always available 
        if 'RAT-1' not in available_rats:
            available_rats.append('RAT-1')
            
        return sorted(available_rats)

    def _get_group_by_rats(self, available_rats: List[str]) -> str:
        """Determine group based on available RATs"""
        available_rats_set = set(available_rats)
        
        for group, required_rats in self.group_definitions.items():
            if set(required_rats) == available_rats_set:
                return group
        
        # Fallback: Find closest matching group
        for group, required_rats in self.group_definitions.items():
            if set(required_rats).issubset(available_rats_set):
                return group
                
        return 'Group D'  # Default to satellite only

    def _generate_coordinates_for_group(self, group: str) -> Tuple[float, float]:
        """Generate coordinates that will result in the desired group, using prioritized ranges"""
        max_attempts = 100  # Prevent infinite loop
        
        for attempt in range(max_attempts):
            # Generate coordinates with prioritization
            lat, lon = self._generate_prioritized_coordinates()
            
            # Check if these coordinates give the desired group
            available_rats = self._get_rats_by_coverage(lat, lon)
            actual_group = self._get_group_by_rats(available_rats)
            
            if actual_group == group:
                return lat, lon
        
        return self.make_coordinatesForGroup(group)

    def make_coordinatesForGroup(self, group: str) -> Tuple[float, float]:
        """
        Return lat, lon that produce the intended RAT-set given:
        RAT-1: lat 23.0-27.6, lon 25.0-35.0
        RAT-4: lat 23.0-27.6, lon 25.0-35.0   (same span as RAT-1)
        RAT-3: lat 23.0-27.6, lon 32.0-35.0
        RAT-5: lat 23.0-27.6, lon 25.0-27.5
        RAT-2: lat 23.0-27.6, lon 28.0-31.0
        """
        
        # Gamma distribution parameters for right-skewed 
        shape = 0.7    
        scale = 2.5
        
        # Generate gamma value and map to latitude range 
        gamma_val = self.rng.gamma(shape, scale)
     
        lat = self.rng.uniform(*self.lat_bounds)
        # Ensure within bounds

        if group == 'Group A':
            # A = {RAT-1, RAT-4, RAT-5} 
            lon = float(self.rng.uniform(25.0, 25.25))  
            return lat, lon

        elif group == 'Group B':
            # B = {RAT-1, RAT-4, RAT-2} 
            lon = float(self.rng.uniform(25.3,25.6))  
            return lat, lon

        elif group == 'Group C':
            # C = {RAT-1, RAT-4} 
            if self.rng.random() < 0.6:
                lon = float(self.rng.uniform(25.6,25.7))  
            else:
                lon = float(self.rng.uniform(25.25,25.3))  
            return lat, lon

        elif group == 'Group D':
            lon = float(self.rng.uniform(25.7,26))
            return lat, lon
            
        else:
            return self._generate_prioritized_coordinates()
