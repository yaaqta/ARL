import pandas as pd
import numpy as np
import env
import dqn
from tqdm import tqdm
import shutil
import os
import json
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# HEMS = Home Energy Management System
class HEMS:
    
    def __init__(self, load=False, path=None, battery=20, max_en=6/4, eff=0.9, price_coefs=[2,1], n_days=2, data_path='data/rtp.csv'):
        self.memory_capacity = 2000
        self.agent = None
        self.path = path
        self.episode_rewards = []
        if load:
            self.load_set_attributes(path)
        else:    
            self.battery = battery
            self.max_en = max_en
            self.eff = eff
            self.price_coefs = price_coefs
            # Kiểm tra nếu 'data/rtp.csv' không tồn tại, tạo dataframe giả định
            if not os.path.exists(data_path):
                print(f"Warning: {data_path} not found. Creating dummy DataFrame.")
                num_steps_dummy = 24 * 4 * 365 # 1 year of 15-min intervals
                self.df = pd.DataFrame({
                    'Timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_steps_dummy, freq='15min')),
                    'RTP': np.random.rand(num_steps_dummy) * 0.2 + 0.1, # giả định giá từ 0.1 đến 0.3
                    'Cons': np.random.rand(num_steps_dummy) * 3 + 1,    # giả định tiêu thụ từ 1 đến 4
                    'Gen': np.random.rand(num_steps_dummy) * 2          # giả định sản xuất từ 0 đến 2
                })
            else:
                self.df = pd.read_csv(data_path)
            
            print("Raw dataset head:")
            print(self.df.head())

            self.epsilon = 1
            self.n_days = n_days # Sử dụng n_days từ constructor
        
        print(f'YOU HAVE INITIALISED YOUR HEMS (Home Energy Management System) WITH FOLLOWING SPECIFICATIONS:\nBattery size: {self.battery} kWh\nMax input/output: {self.max_en} kWh\nEfficiency: {self.eff}\nPrice coefficients: {self.price_coefs}\nNumber of days for training/testing: {self.n_days}')


    def train(self, a=3, b=3, n_episodes=200, epsilon_reduce=0.98, n_days_env=2, n_steps=7*24*4): # Đổi n_days thành n_days_env để tránh nhầm lẫn
        df = self.df
        seed = 0
        # ENVIRONMENT AND AGENT INITIALIZATION
        # Sử dụng n_days_env cho env.Env để có thể điều chỉnh độ dài của môi trường trong quá trình train
        envRL = env.Env(df, self.battery, self.max_en, self.eff, self.price_coefs, n_days=n_days_env, n_steps=n_steps)
        envRL.reset(seed)
    
        self.agent = dqn.DQN(envRL.next_observation_normalized().shape[0], 4)

        # TRAINING
        print('Training in progress...')
        epsilon = self.epsilon
        epsilon_history = [] # List để lưu trữ epsilon qua các episodes
        
        for episode in tqdm(range(n_episodes)):
            epsilon = epsilon * epsilon_reduce
            epsilon_history.append(epsilon) # Lưu epsilon
            
            # run_episode returns [sRL, s_baseline1, s_baseline2, total_en_cost_no_pv_batt, energy_flow_data, actions]
            # but for training, we only care about sRL (reward)
            results = self.run_episode(n_steps, envRL, None, None, self.agent, epsilon, a, b, is_test_mode=False) # Không cần env1, env2 cho train
            cost_dqn = results[0] # sRL is the first element when is_test_mode=False
            self.episode_rewards.append(cost_dqn)
            
            # Print episode results (optional, can be moved to specific intervals)
            print_episode_results(cost_dqn, None, None, f'Episode {episode+1}', epsilon)
            
        self.epsilon = epsilon

        # --- Vẽ biểu đồ reward (đã có và cải tiến) ---
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_rewards, label='Episode Reward', c='blue', alpha=0.7)

        # Tính toán Average reward (ví dụ: trung bình động 50 episode)
        if len(self.episode_rewards) > 50:
            rolling_mean = pd.Series(self.episode_rewards).rolling(window=50).mean()
            plt.plot(rolling_mean, label='Rolling Mean Reward (50 episodes)', c='orange', linewidth=2)

        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.title('The convergence process of the Algorithm')
        plt.grid(True)
        plt.legend()
        plt.savefig('rewards_convergence.png', dpi=300)
        plt.show()
        # ----------------------------------------------------

        # --- Thêm phần vẽ biểu đồ Epsilon Decay ---
        plt.figure(figsize=(10, 6))
        plt.plot(epsilon_history, label='Epsilon value', c='purple')
        plt.xlabel('Episodes')
        plt.ylabel('Epsilon')
        plt.title('Epsilon Decay over Training Episodes')
        plt.grid(True)
        plt.legend()
        plt.savefig('epsilon_decay.png', dpi=300)
        plt.show()
        # ------------------------------------------

    def test(self, a=3, b=3, start=30000, steps=None, title='Test Episode'): # Thêm title
        df = self.df
        seed = 0 # Sử dụng seed cố định cho test để kết quả ổn định
        
        test_n_steps = self.n_days * 24 * 4 if steps is None else steps

        # ENVIRONMENT INITIALIZATION for test and baselines
        envRL_test = env.Env(df, self.battery, self.max_en, self.eff, self.price_coefs, n_days=self.n_days, n_steps=test_n_steps, low=start, test=True)
        env1 = env.Env(df, self.battery, self.max_en, self.eff, self.price_coefs, n_days=self.n_days, n_steps=test_n_steps, low=start, test=True) # Baseline 1
        env2 = env.Env(df, 0, 0, self.eff, self.price_coefs, n_days=self.n_days, n_steps=test_n_steps, low=start, test=True) # No PV/Battery (battery capacity=0, max_energy=0)

        # AGENT INITIALIZATION
        if self.agent is None:
            # Load the agent if not already loaded (e.g., if called directly without previous train)
            self.agent = dqn.DQN(envRL_test.next_observation_normalized().shape[0], 4)
            if self.path:
                self.agent.load(self.path)
            else:
                print("Warning: No agent loaded for testing. Results might be random.")
        
        agent = self.agent
        epsilon = 0 # For testing, usually epsilon is 0 to exploit learned policy

        print('Testing in progress...')
        # run_episode returns [final_cost_dqn, final_cost_baseline1, final_cost_baseline2, total_en_cost_no_pv_batt, energy_flow_data, actions]
        results = self.run_episode(test_n_steps, envRL_test, env1, env2, agent, epsilon, a, b, is_test_mode=True)
        
        cost_dqn = results[0]
        cost_baseline1 = results[1]
        cost_baseline2 = results[2] # This is the final cost for baseline 2
        en_cost_no_pv_batt = results[3] # This is the accumulated cost for No PV/Battery baseline
        energy_flow_data_from_episode = results[4]
        actions_from_episode = results[5]

        # Truyền biến 'title' vào hàm print_episode_results
        print_episode_results(cost_dqn, cost_baseline1, en_cost_no_pv_batt, title, epsilon)

        # --- Vẽ biểu đồ cột so sánh tổng chi phí ---
        labels = ['DQN', 'Baseline 1', 'No PV/Battery']
        costs = [cost_dqn, cost_baseline1, en_cost_no_pv_batt] # Đảm bảo đây là giá trị chi phí dương

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 6))
        rects = ax.bar(x, costs, width, color=['blue', 'red', 'green'])

        ax.set_ylabel('Total Energy Cost ($)')
        ax.set_title('Mean Value of Total Energy Cost Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.tick_params(axis='y')
        ax.set_ylim(bottom=0)

        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate('{}'.format(round(height, 2)),
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects)

        fig.tight_layout()
        plt.savefig('total_energy_cost_bar_chart.png', dpi=300)
        plt.show()
        # ----------------------------------------------------

        # --- Vẽ biểu đồ chi tiết kết quả test (Cumulative Cost, Battery Capacity, Energy Flows) ---
        # Data for plots are already collected within run_episode and returned in results
        # These plots are generated inside run_episode when is_test_mode is True, so no need to duplicate here
        # but if you want to explicitly separate them, you can move plotting code here from run_episode's if env0.test block.
        # For simplicity and to avoid redundant code, I'll keep the test plots within run_episode based on its 'env0.test' flag.
        # So ensure that run_episode's plotting part works correctly.

        # --- Thêm phần vẽ biểu đồ phân bố hành động (tùy chọn) ---
        if actions_from_episode: # Chỉ vẽ nếu có hành động
            plt.figure(figsize=(8, 5))
            action_counts = pd.Series(actions_from_episode).value_counts().sort_index()
            # Đảm bảo tất cả các action có thể có (0, 1, 2, 3) đều được hiển thị ngay cả khi frequency là 0
            all_actions = np.arange(4) # Giả định có 4 hành động (0, 1, 2, 3)
            action_labels = {0: 'Action 0 (Charge max)', 1: 'Action 1 (Discharge max)', 2: 'Action 2 (Balance)', 3: 'Action 3 (No charge/discharge)'} # Tùy chỉnh nếu có mô tả rõ hơn
            
            # Reindex series to include all possible actions and fill missing with 0
            action_counts = action_counts.reindex(all_actions, fill_value=0)
            
            action_counts.plot(kind='bar', color='skyblue')
            plt.xlabel('Action ID')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Chosen Actions by DQN Agent ({title})')
            plt.xticks(all_actions, [f'{idx}' for idx in all_actions], rotation=0) # Chỉ hiển thị ID hành động
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.savefig('action_distribution.png', dpi=300)
            plt.show()
        # --------------------------------------------------------

        return results

    def run_episode(self, n_steps, env0, env1, env2, dqn, epsilon, a, b, is_test_mode=False):
        seed = np.random.randint(0, 1000) if not is_test_mode else 0 # Fixed seed for test mode
        env0.reset(seed)
        if is_test_mode:
            env1.reset(seed) # Reset baseline 1 env
            env2.reset(seed) # Reset baseline 2 env (No PV/Battery)

        state = env0.next_observation_normalized()
        
        sRL = 0 # Cumulative reward for DQN
        s_baseline1 = 0 # Cumulative reward for Baseline 1
        s_baseline2 = 0 # Cumulative reward for Baseline 2 (usually 0 for this type of baseline if it directly calculates cost)

        cumulative_rewardRL = []
        cumulative_reward_baseline1 = []
        cumulative_reward_baseline2 = [] # Cumulative cost for baseline 2 (if applicable)

        en_cost_sums = [] # To accumulate cost for 'No PV/Battery' baseline at each step
        energy_flow_data = [] # To collect detailed energy flow data
        actions = [] # To collect chosen actions

        for step in tqdm(range(n_steps)):
            # Environment 0 (DQN)
            action = dqn.choose_action(state, epsilon)
            actions.append(action) # Collect action
            
            obs, reward, terminated, data = env0.step(action)
            energy_flow_data.append(data) # Collect energy flow data for this step
            
            past_state = state
            state = env0.next_observation_normalized()

            sRL += reward # Accumulate reward for DQN
            cumulative_rewardRL.append(sRL)

            # Environment 1 (Baseline 1)
            if is_test_mode and env1 is not None:
                # Assuming env1 has a method for baseline action or steps directly
                # If env1 needs action: baseline_action_1 = some_baseline_logic(env1.next_observation())
                # If env1 has a dedicated step for baseline:
                obs1, reward1, terminated1, _ = env1.step_baseline1() # Placeholder: ensure this method exists in env.py
                s_baseline1 += reward1
                cumulative_reward_baseline1.append(s_baseline1)

            # Environment 2 (Baseline 2 - No PV, No Battery)
            if is_test_mode and env2 is not None:
                # This env should only calculate cost based on consumption and market price, no battery/PV management
                obs2, reward2, terminated2, _ = env2.step_no_pv_battery() # Placeholder: ensure this method exists in env.py
                s_baseline2 += reward2 # This will be the direct cost for baseline 2 if reward is cost
                cumulative_reward_baseline2.append(s_baseline2)
            
            # Calculate and accumulate cost for "No PV, No Battery" scenario (using env0's current data)
            current_df_row = env0.df.iloc[env0.pos] # Get current data from main env's position
            current_consumption = current_df_row['Cons']
            current_market_price_buy = current_df_row['RTP'] * self.price_coefs[0]
            current_market_price_sell = current_df_row['RTP'] * self.price_coefs[1]
            
            # Cost without PV or battery is simply (consumption * buying price) - (generation * selling price if any)
            # Assuming 'Gen' from df is PV generation
            cost_at_this_step_no_pv_batt = (current_consumption * current_market_price_buy) - (current_df_row['Gen'] * current_market_price_sell)
            en_cost_sums.append(cost_at_this_step_no_pv_batt)

            if terminated:
                break
        
        # Calculate total accumulated cost for 'No PV/Battery' case
        total_en_cost_no_pv_batt = np.sum(en_cost_sums)

        # Plotting for test mode only
        if is_test_mode:
            df = env0.df
            history = np.array(env0.history) # History from DQN's environment
            steps = history[:, 0]
            battery_capacity = history[:, 1]
            # If env0.history stores more data points, you can use them.
            # Example: energy_consumption = history[:, 3]

            timestamp = list(df['Timestamp'][int(steps[0]):int(steps[-1])+1])
            
            fig, ax = plt.subplots(3, 1, figsize=(15, 20)) # 3 rows for plots
            
            ax[0].set_title(f'System specs: battery size={self.battery}kWh, max charge energy={self.max_en}kWh')
            
            # Plot 1: Cumulative Cost
            cumulative_rewardRL = np.array(cumulative_rewardRL)
            cumulative_costRL = - (cumulative_rewardRL - cumulative_rewardRL[0]) # Convert rewards to cumulative costs
            ax[0].plot(steps, cumulative_costRL, label='DQN Cost', c='darkblue', linewidth=0.7)

            # Cumulative cost for 'No PV, No Battery' baseline
            cumulative_en_cost_sums = np.cumsum(en_cost_sums)
            ax[0].plot(steps, cumulative_en_cost_sums, label='No PV, No Battery Cost', c='red', linewidth=0.7)
            ax[0].grid(True)
            
            if env1 is not None and len(cumulative_reward_baseline1) > 0:
                cumulative_reward_baseline1 = np.array(cumulative_reward_baseline1)
                cumulative_cost_baseline1 = - (cumulative_reward_baseline1 - cumulative_reward_baseline1[0])
                # history_baseline1 = np.array(env1.history) # If you want to use env1's specific history
                ax[0].plot(steps, cumulative_cost_baseline1, label='Baseline 1 Cost', c='magenta', linewidth=0.7)
            
            if env2 is not None and len(cumulative_reward_baseline2) > 0:
                cumulative_reward_baseline2 = np.array(cumulative_reward_baseline2)
                cumulative_cost_baseline2 = - (cumulative_reward_baseline2 - cumulative_reward_baseline2[0]) # If baseline 2 also returns rewards
                ax[0].plot(steps, cumulative_cost_baseline2, label='Baseline 2 Cost', c='violet', linewidth=0.7)

            ax[0].legend(loc='upper left', prop={'size': 8})
            ax[0].set_ylabel('Cumulative Cost ($)')
            ax[0].set_xlabel('Time Steps')
            
            # Plot 2: Battery Capacity
            ax[1].plot(steps, battery_capacity, label='DQN Battery Charge', c='blue', linewidth=0.7)
            
            if env1 is not None and len(env1.history) > 0:
                battery_capacity_baseline1 = np.array(env1.history)[:, 1]
                ax[1].plot(steps, battery_capacity_baseline1, label='Baseline 1 Battery Charge', c='magenta', linewidth=0.7, alpha=0.5)
            
            if env2 is not None and len(env2.history) > 0:
                battery_capacity_baseline2 = np.array(env2.history)[:, 1] # This would be 0 for 'No PV/Battery'
                ax[1].plot(steps, battery_capacity_baseline2, label='Baseline 2 Battery Charge', c='violet', linewidth=0.7, alpha=0.5)

            ax[1].legend(loc='lower left', prop={'size': 8})
            ax[1].set_ylabel('Battery Capacity (kWh)')
            ax[1].set_xlabel('Time Steps')
            ax[1].grid(True)

            # Plot 3: Energy Flows
            # energy_flow_data is a list of lists, where each inner list contains:
            # [gen_old, cons_old, e_b_out, e_b_in, e_pv_b, e_pv_grid, e_b_home, e_b_grid, e_grid_home, e_grid_b]
            energy_flow_array = np.array(energy_flow_data)
            
            # Define labels and corresponding column indices for plotting
            flow_labels = [
                'PV Generation (gen_old)', 'Home Consumption (cons_old)',
                'Battery Discharge (e_b_out)', 'Battery Charge (e_b_in)',
                'PV to Battery (e_pv_b)', 'PV to Grid (e_pv_grid)',
                'Battery to Home (e_b_home)', 'Battery to Grid (e_b_grid)',
                'Grid to Home (e_grid_home)', 'Grid to Battery (e_grid_b)'
            ]
            flow_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

            if energy_flow_array.shape[1] > max(flow_indices):
                for idx, label in zip(flow_indices, flow_labels):
                    ax[2].plot(steps, energy_flow_array[:, idx], label=label, linewidth=0.7, alpha=0.8)
                ax[2].set_ylabel('Energy Flow (kWh)')
                ax[2].set_xlabel('Time Steps')
                ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size': 8}) # Place legend outside
                ax[2].grid(True)
            else:
                print("Warning: Energy flow data shape mismatch. Cannot plot all desired flows.")

            fig.tight_layout(rect=[0, 0, 0.88, 1]) # Adjust layout to make space for legend
            fig.savefig('test_results_detailed.png', dpi=300)
            plt.show()
            
            # Convert final rewards to costs for comparison
            final_cost_dqn = -sRL
            final_cost_baseline1 = -s_baseline1 if is_test_mode else 0
            final_cost_baseline2 = -s_baseline2 if is_test_mode else 0 # Ensure this is a cost

            results = [final_cost_dqn, final_cost_baseline1, final_cost_baseline2, total_en_cost_no_pv_batt, energy_flow_data, actions]
        else: # For training, just return the raw rewards/costs
            results = [sRL, s_baseline1, s_baseline2, total_en_cost_no_pv_batt, energy_flow_data, actions]
            
        return results

    def save(self, path):
        # ... (giữ nguyên hàm save) ...
        d = {
            "battery": self.battery,
            "max_en": self.max_en,
            "eff": self.eff,
            "price_coefs": self.price_coefs,
            "epsilon": self.epsilon,
            "n_days": self.n_days, # Lưu n_days
            "episode_rewards": self.episode_rewards # Lưu rewards để có thể tiếp tục plot
        }
        df = self.df
        df.to_csv(path + '/df.csv', index=False) # Lưu df nếu cần
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        self.agent.save(path) # Lưu agent
        
        with open(path + "/properties.json", "w") as outfile:
            json.dump(d, outfile)


    def load_set_attributes(self, path):
        # Load agent
        self.agent = dqn.DQN(n_states=0, n_actions=0) # Initialize dummy agent
        self.agent.load(path) # Load real parameters

        # Load properties
        d = self.load_properties(path)
        self.battery = d['battery']
        self.max_en = d['max_en']
        self.eff = d['eff']
        self.price_coefs = d['price_coefs']
        self.epsilon = d['epsilon']
        self.n_days = d.get('n_days', 2) # Compatible with older saved models
        self.episode_rewards = d.get('episode_rewards', []) # Load episode rewards if available

        # Load DataFrame if saved
        df_path = os.path.join(path, 'df.csv')
        if os.path.exists(df_path):
            self.df = pd.read_csv(df_path)
            print("Loaded DataFrame from:", df_path)
        else:
            print(f"Warning: df.csv not found at {df_path}. Using default data loading path or dummy data.")
            self.df = pd.read_csv('data/rtp.csv') # Fallback to original data path or handle as in __init__

    def load_properties(self, path):
        d = {}       
        with open(path + "/properties.json", "r") as openfile:
            d = json.load(openfile)
        return d
    
def battery_penalty_expand(capacity, full_capacity, zero_low, zero_high):
    x = capacity / full_capacity
    if x < zero_low:
        f = - (2 / zero_low * x - 2) ** 2 / 2
    elif x > zero_high:
        f = - (2 / (1 - zero_high) * (x - zero_high)) ** 2 * 4
    else:
        f = 0
    return f + 1
    
def slope_market_price(capacity, past_capacity, market_price, avg_market_price):
    slope = capacity - past_capacity
    relative_market_price = market_price - avg_market_price
    return - (slope * relative_market_price)

# Sửa đổi print_episode_results để nhận 'title' thay vì 'episode'
def print_episode_results(s1, s2, s3, title, epsilon):
    print('--------------------------')
    print(f'{title} Results:')
    print(f'DQN final score (Cost): {round(s1, 2)}')
    if s2 is not None:
        print(f'Baseline 1 final score (Cost): {round(s2, 2)}')
    if s3 is not None:
        print(f'No PV/Battery final cost: {round(s3, 2)}')
    print(f'Epsilon: {round(epsilon, 2)}')
    print('--------------------------')