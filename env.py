import numpy as np
import pandas as pd # Thêm import pandas

# Tesla powerwall has a power of 5.8kW without sunlight
# therefore, a maximum energy of 5.8 * 1/4 = 1,45kWh can be charged or discharged from the battery
# in a 15 minute interval

class Env:
    def __init__(self, df, full_battery_capacity=20, max_energy=6/4, eff=0.9, price_coefs=[2,1], n_days=2, n_steps=1000, low=0, high=30000, test=False):
        self.amount_paid = None
        self.market_price = None
        self.energy_consumption = None
        self.energy_generation = None
        self.ev_consumption = None
        self.current_battery_capacity = None
        self.time_of_day = None
        self.pos = None
        self.df = df
        self.n_steps = n_steps
        self.window_size = 24 * 4 * n_days
        self.low = low
        self.high = high
        self.test = test

        self.full_battery_capacity = full_battery_capacity
        self.max_energy = max_energy
        self.eff = eff
        self.price_coefs = price_coefs
        self.current_step = 0
        self.history = [] # For storing state history (time, battery, etc.)
        self.reset(0)
        self.state = self.next_observation() # Initial state

    def reset(self, seed):
        np.random.seed(seed)
        self.current_battery_capacity = self.full_battery_capacity / 2 # Start with half battery
        self.pos = self.low + np.random.randint(self.window_size, self.high - self.window_size - self.n_steps) if not self.test else self.low
        self.current_step = 0
        self.history = [] # Clear history on reset

        # Load initial observation
        self.next_observation()

    def next_observation(self):
        # Ensure pos doesn't go out of bounds
        if self.pos >= len(self.df):
            self.pos = len(self.df) - 1 # Prevent index out of bounds on the last step

        current_row = self.df.iloc[self.pos]
        self.energy_generation = current_row['Gen']
        self.energy_consumption = current_row['Cons']
        self.market_price = current_row['RTP']
        self.time_of_day = (self.pos % (24 * 4)) / (24 * 4) # Normalize time of day (0-1)

        # Store history (time, battery_capacity, gen, cons, market_price, amount_paid, time_of_day)
        # Note: self.amount_paid might not be updated yet for the current step,
        # it typically reflects the cost of the *previous* step's action.
        # For initial state, you might want to set it to 0 or infer from initial conditions.
        
        # Ensure history is appended after self.current_battery_capacity and self.amount_paid are updated by step()
        # For next_observation, we don't update amount_paid yet. It's done in step()
        
        self.state = np.array([
            self.current_battery_capacity,
            self.energy_generation,
            self.energy_consumption,
            self.market_price,
            self.amount_paid if self.amount_paid is not None else 0, # Initial amount paid can be 0
            self.time_of_day
        ])
        return self.state
    
    def next_observation_normalized(self):
        # Simple normalization (min-max scaling example)
        # You should define min/max for each state feature based on your dataset
        # For simplicity, let's assume rough ranges
        norm_battery = self.current_battery_capacity / self.full_battery_capacity
        norm_gen = self.energy_generation / self.df['Gen'].max() if self.df['Gen'].max() > 0 else 0
        norm_cons = self.energy_consumption / self.df['Cons'].max() if self.df['Cons'].max() > 0 else 0
        norm_price = self.market_price / self.df['RTP'].max() if self.df['RTP'].max() > 0 else 0
        norm_amount_paid = self.amount_paid / 100 # Assuming max amount paid around 100 for normalization
        norm_time_of_day = self.time_of_day

        return np.array([
            norm_battery, norm_gen, norm_cons, norm_price, norm_amount_paid, norm_time_of_day
        ])

    def step(self, action):
        done = False
        reward = 0
        data_for_plot = [] # To return detailed energy flow for plotting

        p_buy = self.market_price * self.price_coefs[0]
        p_sell = self.market_price * self.price_coefs[1]
        
        gen = self.energy_generation
        cons = self.energy_consumption
        b = self.current_battery_capacity
        b_max = self.max_energy

        e_grid_home = 0
        e_grid_b = 0
        e_b_home = 0
        e_b_grid = 0
        e_pv_b = 0
        e_pv_grid = 0
        e_b_out = 0 # Battery discharge
        e_b_in = 0  # Battery charge

        new_b = b # Initialize new_b with current battery

        # Actions:
        # 0: Charge battery from grid (or maximize PV to battery)
        # 1: Discharge battery to home (or grid if surplus)
        # 2: Balance (try to meet demand from PV/Battery first, then grid)
        # 3: Do nothing/Passthrough (let consumption be met by PV then grid)

        cost = 0

        if action == 0: # Maximize battery charging (from PV or grid)
            # Try to charge battery from PV first
            charge_from_pv = min(gen, (self.full_battery_capacity - b) / self.eff, b_max / self.eff)
            e_pv_b = charge_from_pv
            new_b = b + charge_from_pv * self.eff
            gen_remaining = gen - charge_from_pv

            # If PV not enough or if we want to charge more from grid
            if new_b < self.full_battery_capacity:
                charge_from_grid = min((self.full_battery_capacity - new_b) / self.eff, b_max / self.eff)
                e_grid_b = charge_from_grid
                new_b += charge_from_grid * self.eff

            # Remaining PV goes to home or grid
            if gen_remaining > cons:
                e_pv_home = cons
                e_pv_grid = gen_remaining - cons
            else:
                e_pv_home = gen_remaining
                e_grid_home = cons - gen_remaining

            cost = (e_grid_home + e_grid_b) * p_buy - (e_pv_grid) * p_sell # Cost calculation for this action

        elif action == 1: # Maximize battery discharging
            # Try to meet home consumption from PV first
            if gen >= cons:
                e_pv_home = cons
                e_pv_grid = gen - cons # Excess PV to grid
            else:
                e_pv_home = gen
                remaining_cons = cons - gen
                
                # Discharge battery to meet remaining consumption
                discharge_to_home = min(b, remaining_cons, b_max)
                e_b_home = discharge_to_home
                new_b = b - discharge_to_home
                remaining_cons -= discharge_to_home

                # If still consumption, take from grid
                e_grid_home = remaining_cons

            # If there's surplus battery after meeting home demand, discharge to grid
            if new_b > 0 and e_b_home < b_max: # Check if battery has capacity to discharge more AND didn't max out discharge
                discharge_to_grid = min(new_b, b_max - e_b_home) # Limit by max_energy and remaining battery
                e_b_grid = discharge_to_grid
                new_b -= discharge_to_grid

            cost = (e_grid_home) * p_buy - (e_pv_grid + e_b_grid) * p_sell # Cost calculation

        elif action == 2: # Balance: prioritize self-consumption, then store surplus, then buy for deficit
            # 1. Try to meet home consumption from PV
            if gen >= cons:
                e_pv_home = cons
                gen_remaining = gen - cons
                # Store surplus PV in battery
                charge_from_pv = min(gen_remaining, (self.full_battery_capacity - b) / self.eff, b_max / self.eff)
                e_pv_b = charge_from_pv
                new_b = b + charge_from_pv * self.eff
                e_pv_grid = gen_remaining - charge_from_pv # Remaining PV to grid
            else: # PV not enough for consumption
                e_pv_home = gen
                remaining_cons = cons - gen
                # 2. Use battery to meet remaining consumption
                discharge_to_home = min(b, remaining_cons, b_max)
                e_b_home = discharge_to_home
                new_b = b - discharge_to_home
                remaining_cons -= discharge_to_home
                # 3. Buy from grid for any remaining deficit
                e_grid_home = remaining_cons
            
            cost = (e_grid_home) * p_buy - (e_pv_grid) * p_sell # Cost calculation

        elif action == 3: # Passthrough/Do Nothing (PV to home, then grid for deficit)
            if gen >= cons:
                e_pv_home = cons
                e_pv_grid = gen - cons # Surplus PV to grid
            else:
                e_pv_home = gen
                e_grid_home = cons - gen # Deficit from grid
            new_b = b # Battery capacity remains unchanged
            cost = (e_grid_home) * p_buy - (e_pv_grid) * p_sell # Cost calculation

        # Update battery capacity
        self.current_battery_capacity = max(0, min(new_b, self.full_battery_capacity)) # Ensure within bounds

        # Calculate reward (negative of cost)
        reward = -cost
        self.amount_paid = cost # Update amount paid for the next state observation

        # Prepare data for plotting (energy_flow_data)
        # [gen_old, cons_old, e_b_out, e_b_in, e_pv_b, e_pv_grid, e_b_home, e_b_grid, e_grid_home, e_grid_b]
        # Need to ensure e_b_out/in are correctly calculated for the chosen action
        # For simplicity, I'll calculate them post-action here, assuming e_b_out is total discharged and e_b_in is total charged
        actual_e_b_out = e_b_home + e_b_grid
        actual_e_b_in = e_pv_b + e_grid_b

        data_for_plot = [
            self.energy_generation, # gen_old
            self.energy_consumption, # cons_old
            actual_e_b_out,       # e_b_out (total from battery)
            actual_e_b_in,        # e_b_in (total to battery)
            e_pv_b,               # e_pv_b
            e_pv_grid,            # e_pv_grid
            e_b_home,             # e_b_home
            e_b_grid,             # e_b_grid
            e_grid_home,          # e_grid_home
            e_grid_b              # e_grid_b
        ]

        # Move to next step
        self.current_step += 1
        self.pos += 1
        
        # Check if episode terminated
        if self.current_step >= self.n_steps:
            done = True
        
        # Store history for current state AFTER action
        self.history.append([
            self.pos, # Current DataFrame index
            self.current_battery_capacity,
            cost, # Cost for current step
            self.energy_consumption,
            self.energy_generation,
            self.market_price,
            self.amount_paid,
            self.time_of_day
        ])

        # Get next observation (for the next state)
        next_obs = self.next_observation_normalized()

        return next_obs, reward, done, data_for_plot


    # --- Placeholder Baseline Functions (REQUIRED FOR HEMS.py) ---
    def step_baseline1(self):
        # Implement your Baseline 1 logic here
        # Example: Charge battery when price is low, discharge when high, try to maximize self-consumption
        # This is a simplified rule-based baseline
        done = False
        reward = 0
        data_for_plot = [0]*10 # Dummy data for now

        p_buy = self.market_price * self.price_coefs[0]
        p_sell = self.market_price * self.price_coefs[1]
        
        gen = self.energy_generation
        cons = self.energy_consumption
        b = self.current_battery_capacity
        b_max_charge_discharge = self.max_energy

        new_b = b
        cost = 0

        # Simple rule: if price is low, charge; if price is high, discharge; otherwise, meet demand
        avg_price = self.df['RTP'].mean() # This should be a dynamic average over time window for better baseline

        if self.market_price < avg_price * 0.9: # Low price, try to charge
            charge_amount = min((self.full_battery_capacity - b) / self.eff, b_max_charge_discharge / self.eff)
            new_b = b + charge_amount * self.eff
            cost = charge_amount * p_buy # Cost to charge from grid
            
            # Still need to handle consumption and generation
            if gen >= cons:
                # Surplus PV after charging goes to grid
                cost -= (gen - cons) * p_sell # Selling surplus PV
            else:
                # Buy remaining consumption from grid
                cost += (cons - gen) * p_buy

        elif self.market_price > avg_price * 1.1: # High price, try to discharge
            discharge_amount = min(b, b_max_charge_discharge)
            new_b = b - discharge_amount
            cost = -discharge_amount * p_sell # Income from discharging to grid
            
            # Meet consumption
            if gen + discharge_amount >= cons:
                # Consumption met by PV + battery
                pass
            else:
                # Buy remaining consumption from grid
                cost += (cons - (gen + discharge_amount)) * p_buy

        else: # Normal price, meet consumption with PV, then battery, then grid
            if gen >= cons:
                pass # PV covers consumption, maybe sell surplus (not handled here for simplicity)
            else:
                # Use battery
                discharge_amount = min(b, cons - gen, b_max_charge_discharge)
                new_b = b - discharge_amount
                remaining_cons = (cons - gen) - discharge_amount
                # Buy from grid
                cost = remaining_cons * p_buy
        
        # Update battery capacity for baseline 1
        self.current_battery_capacity = max(0, min(new_b, self.full_battery_capacity))

        reward = -cost
        self.amount_paid = cost # For history tracking

        self.current_step += 1
        self.pos += 1
        if self.current_step >= self.n_steps:
            done = True
        
        # Store history for current state AFTER action for baseline 1
        self.history.append([
            self.pos, # Current DataFrame index
            self.current_battery_capacity,
            cost, # Cost for current step
            self.energy_consumption,
            self.energy_generation,
            self.market_price,
            self.amount_paid,
            self.time_of_day
        ])
        
        return self.next_observation_normalized(), reward, done, data_for_plot # Returns a dummy observation for now

    def step_no_pv_battery(self):
        # Simulate environment with no PV and no battery
        # Cost is simply (consumption * buying price) - (generation * selling price) if you want to consider 'generation' as fixed load reduction
        # In this baseline, 'gen' from dataframe is ignored if it implies PV, and 'battery' is zeroed out.
        done = False
        data_for_plot = [0]*10 # Dummy data

        current_row = self.df.iloc[self.pos]
        current_consumption = current_row['Cons']
        current_generation = 0 # No PV
        current_market_price_buy = current_row['RTP'] * self.price_coefs[0]
        current_market_price_sell = current_row['RTP'] * self.price_coefs[1] # Not applicable if no PV

        cost = current_consumption * current_market_price_buy # Only buying from grid

        reward = -cost
        self.amount_paid = cost

        self.current_step += 1
        self.pos += 1
        if self.current_step >= self.n_steps:
            done = True
            
        # Store history for current state AFTER action for baseline 2 (battery should always be 0)
        self.history.append([
            self.pos, # Current DataFrame index
            0, # Battery capacity is always 0 for this baseline
            cost, # Cost for current step
            current_consumption,
            current_generation, # Always 0 for this baseline
            self.market_price,
            self.amount_paid,
            self.time_of_day
        ])

        return self.next_observation_normalized(), reward, done, data_for_plot # Returns a dummy observation for now


    def action0(self, b, b_max, gen, cons, p_buy, p_sell):
        # Existing action methods (kept for reference, but main step() handles all actions now)
        # You might want to refactor these into the main step() for clarity or remove if not directly called
        pass
    def action1(self, b, b_max, gen, cons, p_buy, p_sell):
        pass
    def action2(self, b, b_max, gen, cons, p_buy, p_sell):
        pass
    def action3(self, b, b_max, gen, cons, p_buy, p_sell):
        pass