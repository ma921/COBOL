import torch
import numpy as np
from openai import OpenAI


def generate_system_prompt():
    text = 'You are assisting me with an optimization task of thermal confort control. \
    We are trying to minimize the computed value at given three conditions we can specify. \
    The target value is computed by the Python library, pythermalcomfort, \
    specifically, pythermalcomfort.models.pmv_ppd.pmv_ppd. \
    This function returns Predicted Mean Vote (PMV) in accordance to main thermal comfort Standards, \
    specifically, the ASHRAE 55 2020. \
    The PMV is an index that predicts the mean value of the thermal sensation votes (self-reported perceptions) \
    of a large group of people on a sensation scale expressed from –3 to +3 \
    corresponding to the categories: cold, cool, slightly cool, neutral, slightly warm, warm, and hot. \
    We take the absolute value of the PMV, so we wish to tune to be neutral. \
    We have three parameters to specify; tdh, rh, and v. \
    tbh is the dry bulb air temperature, of which unit is the Celcius degree, ranging from 12 to 30. \
    rh is the relative humidity, of which unit is the percentage [%], ranging from 30 to 60. \
    v is the average air speed, of which unit is [m/s], ranging from 0 to 1.5.\n'

    code = 'The PMV is computed by the following python code: \n\
    \n\
    """"starting python3 code\n\
    from pythermalcomfort.models import pmv_ppd\n\
    from pythermalcomfort.utilities import v_relative, clo_dynamic\n\
    from pythermalcomfort.utilities import met_typical_tasks\n\
    from pythermalcomfort.utilities import clo_individual_garments\n\
    def _PMV(tdb, rh, v):\n\
        # input variables\n\
        # tdb \in [12, 30]:  mean radiant temperature, [$^{\circ}$C]\n\
        # rh \in [30, 60]: relative humidity, [%]\n\
        # v \in [0, 1.5]: average air speed, [m/s]\n\
        tr = tdb - 1 \n\
        activity = "Typing"  # participant activity description\n\
        garments = ["Sweatpants", "T-shirt", "Shoes or sandals"]\n\
    \n\
        met = met_typical_tasks[activity]  # activity met, [met]\n\
        icl = sum([clo_individual_garments[item] for item in garments])  # calculate total clothing insulation\n\
    \n\
        # calculate the relative air velocity\n\
        vr = v_relative(v=v, met=met)\n\
        # calculate the dynamic clothing insulation\n\
        clo = clo_dynamic(clo=icl, met=met)\n\
    \n\
        # calculate PMV in accordance with the ASHRAE 55 2020\n\
        results = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, standard="ASHRAE")\n\
        # Return the results\n\
        pmv = results["pmv"]\n\
        return abs(pmv) # take absolute error\n\
    \n\
    """""end python3 code\n'
    return text + code

def read_reply_candidates(reply, domain, n=10):
    try:
        train_X_agent = torch.vstack([
            torch.tensor([float(param.split(":")[-1]) for param in l.split("{")[1].split("}")[0].split(",")])
            for l in reply.split("\n")
        ])
    except:
        try:
            train_X_agent = torch.vstack([
                torch.tensor([float(param.split(":")[-1]) for param in l.split("{")[1].split("}")[0].split(",")])
                for l in reply.split("\n")[1:-1]
            ])
        except:
            try:
                train_X_agent = torch.vstack([
                    torch.tensor([float(param.split(":")[-1]) for param in l.split("{")[1].split("}")[0].split(",")])
                    for l in reply.split("\n")[2:-2]
                ])
            except:
                train_X_agent = domain.sample(n)
    return train_X_agent.to(domain.dtype)

def read_reply_classification(reply):
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
    
def read_query_reply(reply):
    try:
        x = [float(param.split(":")[-1]) for param in reply.split("##")[1].split(",")]
    except:
        try:
            x = [float(param.split(":")[-1]) for param in reply.split("##")[-1].split(",")]
        except:
            try:
                x = [float(param.split(":")[-1]) for param in reply.split("##")[-1].split("\n")]
            except:
                try:
                    x = [float(param.split("**")[-1]) for param in reply.split("##")[-1].split("\n")]
                except:
                    try:
                        x = [float(param.split(":")[-1]) for param in reply.split("##")[-2].split(",")]
                    except:
                        try:
                            x = [float(param.split("**")[-1]) for param in reply.split("##")[-2].split(",")]
                        except:
                            x = [float(param.split("**")[-1]) for param in reply.split("\n")[1].split("##")[-2]]
    return x

def safe_reading(reply):
    flag = True
    while flag:
        try:
            x = read_query_reply(reply)
            flag = False
        except:
            print(reply)
            x = 0
            flag = True
    return x

def query_chatgpt(system_prompt, user_prompt, merge=False):
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
    
def configuration(x):
    text = f"parameter configuration: ## "
    for idx, key in enumerate(["tdb", "rh", "v"]):
        text += f"{key}: {x[idx]}"
        if not idx + 1 == 3:
            text +=", "
    text += " ##\n"
    return text

def warmup_prompt(n_init, train_X, train_Y):
    system_prompt = generate_system_prompt()
    user_prompt = f"I’m exploring a subset of parameters. \
    Please suggest {n_init} diverse yet effective configurations \
    to initiate a Bayesian Optimization process for thermal comfort control. \
    You mustn’t include ‘None’ in the configurations. \
    Your response should include only a list of dictionaries, \
    where each dictionary describes one recommended configuration. \
    Do not enumerate the dictionaries. \
    The followings are the examples of observed data, \
    and lower PMV is better and 0 is the best: "
    for x, y in zip(train_X, train_Y):
        user_prompt += configuration(x)
        user_prompt += "PMV: " + str(y.item())
    return system_prompt, user_prompt

def discrimination_prompt(X_next, train_X, train_Y):
    system_prompt = generate_system_prompt()
    user_prompt = f"The performance classification is 0 if the configuration is \
    in the best-performing 25.0% of all configurations, and 1 otherwise. \
    Your response should only contain the predicted performance classification in the format \
    ## Classification: ##.\n"
    for x, y in zip(train_X, train_Y):
        user_prompt += configuration(x)
        user_prompt += f"Classification: {int(y > torch.quantile(train_Y.squeeze(), 0.25))}\n"
    return system_prompt, user_prompt

def llambo_prompt(train_X, train_Y, alpha=0.01, target_value=None):
    if target_value is None:
        target_value = train_Y.min() - alpha * (train_Y.max() - train_Y.min())
    print(target_value)
    
    system_prompt = generate_system_prompt()
    user_prompt = f"Recommend a configuration that can achieve the target performance of {target_value}. \
    Allowable ranges for the parameters are: \n\
    tbh: [12, 30] \n\
    rh: [30, 60] \n\
    v: [0, 1.5] \n\
    Do not recommend values at the minimum or maximum of allowable range, do not recommend rounded values. \
    Recommend values with highest possible float precision. \
    Your response must only contain the predicted configuration, in the format ## configuration ##.\n"
    for x, y in zip(train_X, train_Y):
        user_prompt += configuration(x)
        user_prompt += "PMV: " + str(y.item())
    user_prompt += f"PMV: {target_value}\n"
    user_prompt += f"parameter configuration: "
    return system_prompt, user_prompt


class ThermalConfortPrompt:
    def __init__(self, domain):
        self.domain = domain
        
    def warmup_sampling(self, n_init_agent, train_X, train_Y):
        system_prompt, user_prompt = warmup_prompt(n_init_agent, train_X, train_Y)
        reply = query_chatgpt(system_prompt, user_prompt)
        train_X_agent = read_reply_candidates(reply, self.domain, n=n_init_agent)
        return train_X_agent
    
    def rejection_sampling(self, train_X, train_Y):
        flag = True
        while flag:
            X_next = self.domain.sample(1, qmc=False)
            Y_agent_next = torch.tensor([
                self.classification(X_next.squeeze(), train_X, train_Y)
            ]).to(train_Y)
            if Y_agent_next == 0:
                flag = False
        return X_next

    def classification(self, X_next, train_X, train_Y):
        system_prompt, user_prompt = discrimination_prompt(X_next, train_X, train_Y)
        reply = query_chatgpt(system_prompt, user_prompt)
        y = read_reply_classification(reply)
        return y

    def multiple_classification(self, train_X_agent, train_X, train_Y):
        Y = []
        for X_next in train_X_agent:
            y = self.classification(X_next, train_X, train_Y)
            Y.append(y)
        train_Y_agent = torch.tensor(Y).to(train_X_agent)
        return train_Y_agent

    def initial_sampling(self, n_init_agent, train_X, train_Y):
        train_X_agent = self.warmup_sampling(n_init_agent, train_X, train_Y)
        train_Y_agent = self.multiple_classification(train_X_agent, train_X, train_Y)
        return train_X_agent, train_Y_agent

    def evaluate(self, train_X, train_Y, test_function):
        system_prompt, user_prompt = llambo_prompt(train_X, train_Y)
        reply = query_chatgpt(system_prompt, user_prompt)
        x = torch.tensor(safe_reading(reply)).to(train_X).unsqueeze(0)
        y = test_function(x)
        return x, y
