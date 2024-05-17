import os
import time
import torch
import pandas as pd


class DatasetManager:
    def __init__(self, train_Y, train_Y_agent, global_minimum, save_path, seed):
        self.global_minimum = global_minimum
        self.save_path = save_path
        self.seed = seed
        self.t = 1
        self.t_function = 0
        self.t_agent = 0
        self.delta_t = 0
        self.Y_best = train_Y.min().item()
        self.simple_regret = self.Y_best - self.global_minimum
        self.Y_cumulative = self.simple_regret
        self.N_query = 0
        self.N_query_function = 0
        self.N_init_query = len(train_Y_agent)
        self.N_init_function = len(train_Y)
        self.simple_regrets = [self.simple_regret]
        self.cumulative_regrets = [self.Y_cumulative]
        self.cumulative_queries = [self.N_query]
        self.overheads = [0]
        self.cumulative_agent_queries_global = []
        self.cumulative_function_queries_global = []
        self.t_global = 0
        
    def update_count(self, train_Y, train_Y_agent=None, verbose=False):
        self.t_function = len(train_Y)
        self.t_agent = len(train_Y_agent)
        self.N_query_function = len(train_Y) - self.N_init_function
        
        if train_Y_agent is not None:
            self.N_query = len(train_Y_agent) - self.N_init_query
            if verbose:
                print('Iter: %d N_query_agent: %d best found: %.3f ' % (self.t, self.N_query, self.Y_best))
        else:
            if verbose:
                print('Iter: %d best found: %.3f ' % (self.t, self.Y_best))
        
    def tik(self, train_Y, train_Y_agent):
        self.t_global += 1
        self.t_function = len(train_Y)
        self.t_agent = len(train_Y_agent)
        
        self.update_count(train_Y, train_Y_agent, verbose=False)
        self.cumulative_agent_queries_global.append(self.N_query)
        self.cumulative_function_queries_global.append(self.N_query_function)
        log = pd.DataFrame(
             {
                 't': torch.arange(self.t_global).tolist(),
                 'cumulative_agent_queries': self.cumulative_agent_queries_global,
                 'cumulative_function_queries': self.cumulative_function_queries_global,
                 'seed': [int(self.seed)] * self.t_global,
             }
        )
        log.to_csv(os.path.join(self.save_path, f"log_queries_seed{self.seed}.csv"))
        self.start_time = time.monotonic()
        
    def tok(self, train_X, train_Y, X_next, Y_next, train_Y_agent=None):
        end_time = time.monotonic()
        self.overhead = end_time - self.start_time
        self.overheads.append(self.overhead)
        
        train_X, train_Y = self.cat_dataset(train_X, train_Y, X_next, Y_next)
        self.update_record(train_Y, train_Y_agent)
        log = self.save_log()
        return train_X, train_Y, log
        
    def update_record(self, train_Y, train_Y_agent=None):
        self.Y_best = train_Y.min().item()
        self.simple_regret = self.Y_best - self.global_minimum
        self.Y_cumulative += self.simple_regret
        self.simple_regrets.append(self.simple_regret)
        self.cumulative_regrets.append(self.Y_cumulative)
        self.update_count(train_Y, train_Y_agent, verbose=True)
        self.cumulative_queries.append(self.N_query)
        self.t += 1
    
    def save_log(self):
        log = pd.DataFrame(
             {
                 't': torch.arange(1,self.t+1).tolist(),
                 'simple_regrets': self.simple_regrets,
                 'cumulative_y': self.cumulative_regrets,
                 'cumulative_queries': self.cumulative_queries,
                 'overhead': self.overheads,
                 'seed': [int(self.seed)] * len(self.simple_regrets),
             }
        )
        log.to_csv(os.path.join(self.save_path, f"log_seed{self.seed}.csv"))
        return log
        
    def cat_dataset(self, train_X, train_Y, X_next, Y_next):
        train_X = torch.cat([train_X, X_next])
        train_Y = torch.cat([train_Y, Y_next])
        return train_X, train_Y
