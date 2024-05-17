import torch
import numpy as np
import scipy.stats
from openai import OpenAI


class Interpret:
    def read_reply(self, reply):
        try:
            x = [float(param.split(":")[-1]) for param in reply.split("##")[1].split(",")]
        except:
            try:
                x = [float(param.split(":")[-1]) for param in reply.split("##")[-1].split(",")]
            except:
                try:
                    x = [float(param.split(":")[-1]) for param in reply.split("##")[-1].split("\n")]
                except:
                    x = [float(param.split("**")[-1]) for param in reply.split("##")[-1].split("\n")]
        return x
    
    def transform(self, x):
        return torch.tensor([
            self.transform_each(x[idx], value) for idx, value in enumerate(self.hypers.values())
        ])
    
    def read_reply_candidates(self, reply, n=10):
        try:
            train_X_agent = torch.vstack([
                self.transform([float(param.split(":")[-1]) for param in l.split("{")[1].split("}")[0].split(",")])
                for l in reply.split("\n")
            ])
        except:
            try:
                train_X_agent = torch.vstack([
                    self.transform([float(param.split(":")[-1]) for param in l.split("{")[1].split("}")[0].split(",")])
                    for l in reply.split("\n")[1:-1]
                ])
            except:
                try:
                    train_X_agent = torch.vstack([
                        self.transform([float(param.split(":")[-1]) for param in l.split("{")[1].split("}")[0].split(",")])
                        for l in reply.split("\n")[2:-2]
                    ])
                except:
                    train_X_agent = self.domain.sample(n)
        return train_X_agent
    
    def read_reply_classification(self, reply):
        try:
            y = int(reply)
        except:
            try:
                y = int(reply.split("##")[1].split(":")[-1])
            except:
                try:
                    y = int(reply.split("##")[1].split("classification")[-1])
                except:
                    try:
                        y = int(reply.split(":")[-1])
                    except:
                        try:
                            y = int(reply.split("\n")[0].split(":")[-1])
                        except:
                            try:
                                y = int(reply.split(".")[0].split(":")[-1])
                            except:
                                print(reply)
                                y = int(torch.bernoulli(torch.tensor(0.5)).item())
        return y
    
    def read_reply_prediction(self, reply):
        try:
            y = float(reply)
        except:
            try:
                y = float(reply.split("##")[1])
            except:
                try:
                    y = float(reply.split("##")[1].split(":")[-1])
                except:
                    try:
                        y = float(reply.split("##")[1].split("performance")[-1])
                    except:
                        try:
                            y = float(reply.split(":")[-1])
                        except:
                            try:
                                y = float(reply.split("\n")[0].split(":")[-1])
                            except:
                                try:
                                    y = float(reply.split(".")[0].split(":")[-1])
                                except:
                                    y = torch.rand(1).item()
        return y
    
    
    
    def read(self, reply, mode, n=None):
        if mode == "query":
            x = self.read_reply(reply)
        elif mode == "warmup":
            x = self.read_reply_candidates(reply, n=n)
        elif mode == "classification":
            x = self.read_reply_classification(reply)
        elif mode == "prediction":
            x = self.read_reply_prediction(reply)
        return x

class WarmupPrompt:
    def hyperparameter_description(self):
        text = f"The numbder of hyperparameters: {len(self.hypers)}\n"
        for idx, (key, value) in enumerate(zip(self.hypers.keys(), self.hypers.values())):
            text += f"{idx+1}-th hyperparameter: {key}, data type: {value[0]}, transform: {value[1]}, bounds: {value[2]}\n"
        return text

    def configuration(self, x, y):
        text = f"Performance: {y}\n"
        text += f"Hyperparameter configuration: ## "

        d = self.config(x)
        for idx, key in enumerate(d.keys()):
            text += f"{key}: {d[key]}"
            if not idx + 1 == len(self.hypers):
                text +=", "
        text += " ##\n"
        return text

    def warmup_prompt(self, n_init, train_X, train_Y):
        system_prompt = f"You are assisting me with automated machine learning using {self.model} for a {self.task} task. "
        system_prompt += f"The {self.task} performance is measured using {self.metric}. "
        system_prompt += self.dataset_description()
        system_prompt += self.statistical_description()
        user_prompt = f"I’m exploring a subset of hyperparameters detailed as:\n"
        user_prompt += self.hyperparameter_description()
        user_prompt += f"Please suggest {n_init} diverse yet effective configurations \
        to initiate a Bayesian Optimization process for hyperparameter tuning. \
        You mustn’t include ‘None’ in the configurations. \
        Your response should include only a list of dictionaries, \
        where each dictionary describes one recommended configuration. Do not enumerate the dictionaries."
        user_prompt += f"The followings are the examples of observed data, and lower performance value is better: "
        for x, y in zip(train_X, train_Y):
            user_prompt += self.configuration(x, y)
        return system_prompt, user_prompt

class LLAMBOPrompt:
    def dataset_description(self):
        prompt = f"The dataset has {self.n_samples} samples with {self.n_dims} total features, "
        prompt += f"of which {self.dataset_exp[0]} are numerical and {self.dataset_exp[1]} are categorical. "
        prompt += f"Class distribution is {self.dataset_exp[2]}."
        return prompt
    
    def statistical_description(self):
        X = self.dataset["train_x"]
        Y = self.dataset["train_y"]
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        Y_std = (Y - Y.mean(axis=0)) / Y.std(axis=0)
        skewness = np.nan_to_num(scipy.stats.skew(X_std))
        corr_xy = np.array([np.corrcoef(x_std, Y_std)[0,1] for x_std in X_std.T])
        n_strong_corr_xy = (np.abs(corr_xy) > 0.5).sum()
        corr_xx = np.corrcoef(X_std.T)
        n_X, _ = corr_xx.shape
        n_pariwise = n_X * (n_X - 1)
        n_strong_corr_xx = (np.abs(corr_xx) > 0.5).sum() - n_X
        
        prompt = f"We are standarizing numerical values to have mean 0 and std 1. "
        prompt += f"The Skewness of each feature is {skewness}. "
        prompt += f"The number of features that have strong correlation (defined as > 0.5 or <-0.5) "
        prompt += f"with the target feature is {n_strong_corr_xy}. "
        prompt += f"Of the {n_pariwise} pairwise feature relationships, "
        prompt += f"{n_strong_corr_xx} pairs of features are strongly correlated (>0.5, <-0.5)."
        return prompt
    
    def llambo_prompt(self, train_X, train_Y, alpha=0.01, target_value=None):
        if target_value is None:
            target_value = train_Y.min() - alpha * (train_Y.max() - train_Y.min())
        print(target_value)
        
        system_prompt = f"The following are examples of the performance of a {self.model} measured in {self.metric} "
        system_prompt += f"and the corresponding model hyperparameter configurations. "
        system_prompt += f"The model is evaluated on a tabular {self.task} task containing {self.dataset_exp[-1]} classes."
        system_prompt += self.dataset_description()
        user_prompt = f"The allowable ranges for the hyperparameters are:\n"
        user_prompt += self.hyperparameter_description()
        user_prompt += f"Recommend a configuration that can achieve the target performance of {target_value}. \
        Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. \
        Recommend values with highest possible precision, as requested by the data type. \
        Your response must only contain the predicted configuration, in the format ## configuration ##.\n"
        for x, y in zip(train_X, train_Y):
            user_prompt += self.configuration(x, y)
        user_prompt += f"Performance: {target_value}\n"
        user_prompt += f"Hyperparameter configuration: "
        return system_prompt, user_prompt

class DiscreminatorPrompt:
    def discrimination_configuration(self, x):
        text = f"Hyperparameter configuration: ## "
        d = self.config(x)
        for idx, key in enumerate(d.keys()):
            text += f"{key}: {d[key]}"
            if not idx + 1 == len(self.hypers):
                text +=", "
        text += " ##\n"
        return text
    
    def discrimination_prompt(self, X_next, train_X, train_Y):
        system_prompt = f"The following are examples of the performance of a {self.model} measured in {self.metric} "
        system_prompt += f"and the corresponding model hyperparameter configurations. "
        system_prompt += f"The model is evaluated on a tabular {self.task} task containing {self.dataset_exp[-1]} classes."
        system_prompt += self.dataset_description()
        user_prompt = f"The performance classification is 0 if the configuration is \
        in the best-performing 25.0% of all configurations, and 1 otherwise. \
        Your response should only contain the predicted performance classification in the format \
        ## Classification: ##.\n"
        for x, y in zip(train_X, train_Y):
            user_prompt += self.discrimination_configuration(x.squeeze())
            user_prompt += f"Classification: {int(y > torch.quantile(train_Y.squeeze(), 0.25))}\n"
        user_prompt += self.discrimination_configuration(X_next.squeeze())
        user_prompt += f"Classification: "
        return system_prompt, user_prompt
    
class PredictivePrompt:
    def predictive_prompt(self, X_next, train_X, train_Y):
        system_prompt = f"The following are examples of the performance of a {self.model} measured in {self.metric} "
        system_prompt += f"and the corresponding model hyperparameter configurations. "
        system_prompt += f"The model is evaluated on a tabular {self.task} task containing {self.dataset_exp[-1]} classes."
        system_prompt += self.dataset_description()
        user_prompt = f"Your response should only contain the predicted accuracy in the format ## performance ##."
        for x, y in zip(train_X, train_Y):
            user_prompt += self.discrimination_configuration(x)
            user_prompt += f"Performace: {y}\n"
        user_prompt += self.discrimination_configuration(X_next.squeeze())
        user_prompt += f"Performace: "
        return system_prompt, user_prompt
    
    def LCB_MC_approx(self, X_next, train_X, train_Y, n_mc=10):
        Y = torch.tensor([self.prediction(X_next, train_X, train_Y) for _ in range(n_mc)]).to(train_X)
        LCB = Y.mean() - Y.std()
        return LCB

    def query_LCB(self, X_cand, train_X, train_Y, n_mc=10):
        LCB = torch.tensor([self.LCB_MC_approx(X_next, train_X, train_Y, n_mc=n_mc) for X_next in X_cand])
        X_next = X_cand[LCB.argmin()]
        return X_next
    
class SamplingHelper:
    def query_chatgpt(self, system_prompt, user_prompt, merge=False):
        client = OpenAI()
        if merge:
            message=[
                {"role": "user", "content": system_prompt + " " + user_prompt}
            ]
        else:
            message=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        completion = client.chat.completions.create(
          model="gpt-3.5-turbo",
          messages=message,
        )
        return completion.choices[0].message.content

    def safe_reading(self, reply, mode="query", n=None):
        flag = True
        while flag:
            try:
                x = self.read(reply, mode, n=n)
                flag = False
            except:
                print(reply)
                x = 0
                flag = True
        return x  
    
    def warmup_sampling(self, n_init_agent, train_X, train_Y):
        system_prompt, user_prompt = self.warmup_prompt(n_init_agent, train_X, train_Y)
        reply = self.query_chatgpt(system_prompt, user_prompt)
        train_X_agent = self.safe_reading(reply, mode="warmup", n=n_init_agent)
        return train_X_agent
    
    def rejection_sampling(self, train_X, train_Y, domain):
        flag = True
        while flag:
            X_next = domain.sample(1, qmc=False)
            Y_agent_next = torch.tensor([self.classification(X_next.squeeze(), train_X, train_Y)]).to(train_Y)
            if Y_agent_next == 0:
                flag = False
        return X_next
    
    def multiple_classification(self, train_X_agent, train_X, train_Y):
        Y = []
        for X_next in train_X_agent:
            y = self.classification(X_next, train_X, train_Y)
            Y.append(y)
        train_Y_agent = torch.tensor(Y).to(train_X_agent)
        return train_Y_agent

class ExplorationPrompt(Interpret, WarmupPrompt, LLAMBOPrompt, DiscreminatorPrompt, PredictivePrompt, SamplingHelper):
    def __init__(self, test_function):
        self.model = test_function.benchmark.model
        self.task = test_function.benchmark.task
        self.metric = test_function.benchmark.metric
        self.n_samples, self.n_dims = test_function.benchmark.dataset["train_x"].shape
        self.hypers = test_function.benchmark.hyperparameter_constraints
        self.config = test_function.convert_to_config
        self.eval = test_function.evaluate
        self.global_minimum = test_function.global_minimum
        self.transform_each = test_function.transform
        self.dataset_exp = [test_function.n_continuous, test_function.n_categorical, test_function.class_dist, test_function.n_targets]
        self.dataset = test_function.benchmark.dataset
        self.domain = test_function.setup_domain()
    
    def initial_sampling(self, n_init_agent, train_X, train_Y):
        train_X_agent = self.warmup_sampling(n_init_agent, train_X, train_Y)
        train_Y_agent = self.multiple_classification(train_X_agent, train_X, train_Y)
        return train_X_agent, train_Y_agent
    
    def generative_sampling(self, n_cand, domain, train_X, train_Y):
        X_cand = torch.vstack([
            self.rejection_sampling(train_X, train_Y, domain) for _ in range(n_cand)
        ])
        return X_cand
    
    def evaluate(self, train_X, train_Y):
        system_prompt, user_prompt = self.llambo_prompt(train_X, train_Y)
        reply = self.query_chatgpt(system_prompt, user_prompt)
        x = self.safe_reading(reply, mode="query")
        y = self.eval(x, transform=False)
        x = self.transform(x).to(train_X).unsqueeze(0)
        y = torch.tensor([y]).to(train_X)
        return x, y

    def classification(self, X_next, train_X, train_Y):
        system_prompt, user_prompt = self.discrimination_prompt(X_next, train_X, train_Y)
        reply = self.query_chatgpt(system_prompt, user_prompt)
        y = self.read_reply_classification(reply)
        return y
    
    def prediction(self, X_next, train_X, train_Y):
        system_prompt, user_prompt = self.predictive_prompt(X_next, train_X, train_Y)
        reply = self.query_chatgpt(system_prompt, user_prompt)
        y = self.safe_reading(reply, mode="prediction")
        return y

    def mc_query(self, domain, train_X, train_Y, n_cand=5, n_mc=10):
        print("generate candidates....")
        X_cand = self.generative_sampling(n_cand, domain, train_X, train_Y)
        print("MC approaximation of LCB....")
        X_next = self.query_LCB(X_cand, train_X, train_Y, n_mc=n_mc)
        return X_next.unsqueeze(0)
