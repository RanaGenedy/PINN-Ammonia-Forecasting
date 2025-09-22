# pbm_module.py
import yaml
import pandas as pd
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from typing import List, Tuple, Dict

class PROCESS_BASED_MODEL:
    """
    A class to encapsulate the Process-Based Model (PBM) for simulating
    ammonia emissions from manure storage.
    """
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initializes the PBM_computations class by loading constants from a config file.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self._load_constants()

    def _load_constants(self):
        """Loads and organizes constants from the config dictionary."""
        self.sim_consts = self.config['simulation_constants']
        self.herd_consts = self.config['herd_manure_management']
        self.storage_dims = self.config['storage_dimensions']
        self.thermal_props = self.config['thermal_properties']
        self.mineral_props = self.config['mineralization_properties']
        self.evap_props = self.config['evaporation_properties']

    def run_simulation(
        self,
        x_data: np.ndarray,
        diff_params: float,
        k0_n2o: float,
        init_on: float,
        ph: float
    ) -> np.ndarray:
        """
        Runs the full process-based model simulation for a batch of input data.
        """
        # Correctly map all input columns
        df_x = pd.DataFrame(x_data, columns=['AAT', 'WS', 'WD', 'SR', 'RH', 'RF', 'Agitation'])
        batch_size = len(df_x)

        # --- Initial Conditions ---
        mON_new = init_on
        mTAN_new = 0.78 * mON_new
        Told = 5.0 + 0.75 * df_x["AAT"].iloc[0]
        total_manure_depth = self.storage_dims['Depth0']

        # --- Pre-calculate constant daily inputs and storage area ---
        daily_manure_in_kg = self.herd_consts['NAU'] * self.herd_consts['MW']
        daily_bedding_in_kg = self.herd_consts['NAU'] * self.herd_consts['B']
        daily_water_in_kg = self.herd_consts['WW']
        storage_area = self.calculate_storage_area(self.storage_dims, self.herd_consts)

        # --- Simulation Loop ---
        nh3_flux_results = np.zeros((batch_size, 1))
        manure_removal_amts = self._manure_removal_events(df_x)

        for n in range(batch_size):
            # 1. Update Manure Depth based on inputs, outputs, and weather
            weather_data = {"T_air": df_x["AAT"][n], "RH": df_x["RH"][n], "Wind": df_x["WS"][n]}
            daily_evap_kg = self._calculate_daily_evaporation(self.evap_props, weather_data, storage_area)
            
            daily_inputs = {
                "manure_in_kg": daily_manure_in_kg, "bedding_in_kg": daily_bedding_in_kg,
                "water_in_kg": daily_water_in_kg, "precip_m": df_x["RF"][n] / 1000, # convert mm to m
                "water_evap_kg": daily_evap_kg, "removal_percent": manure_removal_amts[n],
                "surface_area_m2": storage_area
            }
            new_layer_depth, total_manure_depth = self._update_manure_depth(
                self.herd_consts, daily_inputs, total_manure_depth, self.sim_consts['dz']
            )

            # 2. Calculate Boundary Temperatures
            T_surface_n = 5.0 + 0.75 * df_x["AAT"][n]
            T_bot_n = self.soil_temperature(self.storage_dims['Soildepth'], n, self.thermal_props['soilparam'])

            # 3. Solve Heat Transfer
            T, nz, nz_new = self.solve_1d_heat_transfer(
                simulation_step=n, total_height=total_manure_depth, new_layer_height=new_layer_depth,
                dz=self.sim_consts['dz'], thermal_params=self.thermal_props['Thermalparam'],
                bottom_temp=T_bot_n, surface_temp=T_surface_n, dt=self.sim_consts['dt'],
                internal_time_steps=24, prev_temp_profile=Told
            )
            Told = T

            # 4. Calculate Nitrogen Mineralization
            if n == 0:
                mTAN_old, mON_old = mTAN_new, mON_new
            mineralization_params = self.mineral_props['Mineralizationparam'] + [k0_n2o]
            mTAN, mON = self.manure_tan_mineralization(
                mTAN_old, mTAN_new, T, mON_old, mON_new, nz, nz_new, mineralization_params
            )
            mON_old, C_old = mON, mTAN

            # 5. Solve Mass Transfer (Diffusion)
            C, _ = self.solve_1d_mass_transfer(
                total_height=total_manure_depth, dz=self.sim_consts['dz'],
                diffusion_coeff=diff_params * 1e-9, prev_conc_profile=C_old,
                surface_flux=self.sim_consts['J'], dt=self.sim_consts['dt'], internal_time_steps=24
            )
            mTAN_old = C.reshape(nz, 1)

            # 6. Calculate Ammonia Emission
            nh3_em, _, _ = self.calculate_ammonia_emission(
                liquid_temp=T[-1][0], air_temp=df_x["AAT"][n], pressure=self.sim_consts['P'], ph=ph,
                tan_conc_surface=C[-1][0], wind_speed=df_x["WS"][n],
                wind_measure_h=self.sim_consts['WindH'], roughness_len=self.sim_consts['z0'],
                air_nh3_conc=self.sim_consts['CAIR']
            )
            nh3_flux_results[n] = nh3_em * 1e6 * 86400  # Convert kg/m^2/s to mg/m^2/d

        return nh3_flux_results.astype('float32')

    def _manure_removal_events(self, df_x: pd.DataFrame) -> List[int]:
        """Determines manure removal events from agitation data."""
        return [90 if ag == 1 else 0 for ag in df_x['Agitation']]

    # =========================================================================
    #  PHYSICS-BASED SUB-MODELS (Implemented & Refactored)
    # =========================================================================

    def calculate_storage_area(self, storage_dims: Dict, herd_consts: Dict) -> float:
        """Calculates the required surface area of the manure storage tank."""
        s_days = storage_dims['sdays']
        # Calculate total volumes over the storage period
        runoff_volume_m3 = storage_dims['S25y'] * storage_dims['Runoffarea']
        bedding_volume_m3 = 0.3 * (herd_consts['NAU'] * herd_consts['B'] * s_days) / herd_consts['BD']
        wash_water_volume_m3 = (herd_consts['WW'] * s_days) / herd_consts['WD']
        manure_volume_m3 = (herd_consts['MW'] * herd_consts['NAU'] * s_days) / herd_consts['MD']

        total_storage_volume_m3 = runoff_volume_m3 + bedding_volume_m3 + wash_water_volume_m3 + manure_volume_m3
        
        # Calculate the available height for manure accumulation
        manure_height = storage_dims['Totalheight'] - storage_dims['Depth0'] - storage_dims['Rain'] - storage_dims['Freeboard']
        
        surface_area_m2 = total_storage_volume_m3 / manure_height
        return surface_area_m2

    def _calculate_daily_evaporation(self, evap_props: Dict, weather_data: Dict, surface_area: float) -> float:
        """Calculates total water evaporation from the surface for one day."""
        a, h0, Ce = evap_props['a'], evap_props['h0'], evap_props['Ce']
        
        # Correct wind speed to standard height
        wind_speed_corr = weather_data['Wind'] * (2 / h0)**a
        
        # Estimate surface temperature
        T_surface = 5.0 + weather_data['T_air'] * 0.75
        # Calculate vapor pressures [kPa]
        e_Ta = 0.61078 * np.exp((weather_data['T_air'] * 17.269) / (237.3 + weather_data['T_air']))
        # Actual vapor pressure of air
        e_a = weather_data['RH'] * e_Ta / 100
        # Saturation vapor pressure at surface temperature
        e_s = 0.61078 * np.exp((17.269 * T_surface) / (237.3 + T_surface))
        
        # Evaporation flux [kg/m^2/s]
        evap_flux = (0.622 / (287.04 * (T_surface + 273.15))) * (e_s - e_a) * wind_speed_corr * Ce
        
        # Total evaporated water per day [kg/day]
        total_evap_kg_day = evap_flux * surface_area * 86400
        return total_evap_kg_day

    def _update_manure_depth(self, herd_consts: Dict, daily_inputs: Dict, prev_total_depth: float, dz: float) -> Tuple[float, float]:
        """Calculates the change in manure depth for one day."""
        # Calculate net volume change [m^3/day]
        manure_vol = daily_inputs['manure_in_kg'] / herd_consts['MD']
        bedding_vol = daily_inputs['bedding_in_kg'] / herd_consts['BD']
        water_in_vol = daily_inputs['water_in_kg'] / herd_consts['WD']
        precip_vol = daily_inputs['precip_m'] * daily_inputs['surface_area_m2']
        evap_vol = daily_inputs['water_evap_kg'] / herd_consts['WD']

        net_vol_change = manure_vol + bedding_vol + water_in_vol + precip_vol - evap_vol
        
        # Account for manure removal
        vol_after_removal = net_vol_change * (1 - daily_inputs['removal_percent'] / 100)
        
        # Calculate change in height [m/day]
        depth_change = vol_after_removal / daily_inputs['surface_area_m2']
        
        # Round to the nearest discrete layer thickness
        new_layer_depth = round(depth_change / dz) * dz
        new_total_depth = prev_total_depth + new_layer_depth
        
        return new_layer_depth, new_total_depth

    def solve_1d_heat_transfer(self, simulation_step: int, total_height: float, new_layer_height: float, dz: float, thermal_params: List[float], bottom_temp: float, surface_temp: float, dt: float, internal_time_steps: int, prev_temp_profile: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Solves the 1D heat transfer equation using an implicit finite difference method."""
        # (Implementation from previous response)
        thermal_conductivity, density, heat_capacity, heat_generation = thermal_params
        num_nodes = round(total_height / dz) + 1
        new_layer_nodes = round(new_layer_height / dz)
        s = (thermal_conductivity * dt) / ((dz ** 2) * density * heat_capacity)
        main_diag = np.full(num_nodes, 1 + 2 * s)
        off_diag = np.full(num_nodes - 1, -s)
        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(num_nodes, num_nodes), format='csr')
        A[0, 0], A[0, 1] = 1, 0
        A[-1, -1], A[-1, -2] = 1, 0

        temp_profile = np.zeros((num_nodes, 1))
        if simulation_step == 0:
            temp_profile[:] = surface_temp
        else:
            old_nodes = num_nodes - new_layer_nodes
            if old_nodes > 0 and len(prev_temp_profile) >= old_nodes:
                temp_profile[0:old_nodes] = prev_temp_profile[:old_nodes]
            temp_profile[old_nodes:num_nodes] = surface_temp

        heat_gen_term = (heat_generation / (density * heat_capacity)) * dt
        for _ in range(internal_time_steps):
            rhs = temp_profile + heat_gen_term
            rhs[0], rhs[-1] = bottom_temp, surface_temp
            temp_profile = spsolve(A, rhs).reshape(-1, 1)
        return temp_profile, num_nodes, new_layer_nodes
        
    def solve_1d_mass_transfer(self, total_height: float, dz: float, diffusion_coeff: float, prev_conc_profile: np.ndarray, surface_flux: float, dt: float, internal_time_steps: int) -> np.ndarray:
        """Solves the 1D mass transfer (diffusion) equation."""
        num_nodes = round(total_height / dz) + 1
        s = (diffusion_coeff * dt) / (dz ** 2)
        
        # Construct the coefficient matrix A 
        main_diag = np.full(num_nodes, 1 + 2 * s)
        off_diag = np.full(num_nodes - 1, -s)
        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(num_nodes, num_nodes), format='csr')
        
        # Boundary Conditions
        A[0, 0], A[0, 1] = 1 + 2 * s, -2 * s  # Bottom flux condition (zero flux)
        A[-1, -1], A[-1, -2] = 1 + 2 * s, -2 * s # Surface flux condition
        
        conc_profile = np.copy(prev_conc_profile).reshape(-1, 1)
        
        for _ in range(internal_time_steps):
            rhs = np.copy(conc_profile) # Use a copy to avoid modification issues
            rhs[0] = conc_profile[0] - (0 * 2 * s * dz / diffusion_coeff) # Bottom flux is 0
            rhs[-1] = conc_profile[-1] + (surface_flux * 2 * s * dz / diffusion_coeff)
            conc_profile = spsolve(A, rhs).reshape(num_nodes, 1) # Reshape correctly
            
        return conc_profile, None # Return None for Cdata

    def manure_tan_mineralization(self, mTAN_old, mTAN_new, temp_profile, mON_old, mON_new, nz, nz_new, mineralization_params):
        """Calculates TAN concentration after organic nitrogen mineralization."""
        theta, kON20 = mineralization_params
        mON = np.zeros((nz, 1))
        TAN_total = np.zeros((nz, 1))

        # Combine old and new profiles
        mON[0:(nz - nz_new)] = mON_old
        mON[(nz - nz_new):nz] = mON_new
        TAN_total[0:(nz - nz_new)] = mTAN_old
        TAN_total[(nz - nz_new):nz] = mTAN_new

        # Calculate mineralization
        kON = kON20 * (theta ** (temp_profile - 20))
        mTAN_generated = kON * mON
        mTAN_final = TAN_total + mTAN_generated
        mON_final = mON - mTAN_generated
        return mTAN_final, mON_final

    def soil_temperature(self, soil_depth: float, time_day: int, soil_params: Dict) -> float:
        """Calculates soil temperature at a given depth and time."""
        avg_temp = soil_params['annual_avg_temp']
        amp = soil_params['annual_amp']
        diffusivity = soil_params['thermal_diffusivity']
        time_lag = soil_params.get('time_lag_days', 0) # Day of min surface temp
        omega = 2 * np.pi / 365.0
        
        # Calculate damping depth, which indicates how deep temperature fluctuations penetrate
        damping_depth = (2 * diffusivity / omega)**(1/2)
        term1 = amp * np.exp(-soil_depth / damping_depth)
        term2 = np.sin((2 * np.pi * (time_day - time_lag) / 365) - (soil_depth / damping_depth) - (np.pi / 2))
        T_bot = avg_temp + term1 * term2
        return T_bot

    def calculate_ammonia_emission(self, liquid_temp, air_temp, pressure, ph, tan_conc_surface, wind_speed, wind_measure_h, roughness_len, air_nh3_conc):
        """Calculates the NH3 emission flux from the manure surface."""
        T_liq_K = liquid_temp + 273.15
        T_air_K = air_temp + 273.15

        U_8m = wind_speed * (np.log(8 / roughness_len)) / np.log(wind_measure_h / roughness_len)
        Ka = 10**(0.09018 - (2729.92 / T_liq_K)) # Dissociation constant
        F = 1 / (1 + (10**(-ph) / Ka)) # Fraction of free ammonia

        kL = (1.6761e-6) * np.exp(0.236 * U_8m) # Liquid phase mass transfer coeff
        kG = (5.1578e-5) + (2e-3 * U_8m) # Gas phase mass transfer coeff
        H = (1 / (1.64E-04 * np.exp(1038 / T_liq_K))) # Henry's constant (dimensionless)
        
        KL = 1 / ((1 / kL) + (1 / (H * kG))) # Overall mass transfer coefficient [m/s]
        nh3_flux = KL * (F * tan_conc_surface - air_nh3_conc) # [kg/m^2/s]
        return nh3_flux, Ka, F