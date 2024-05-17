import torch
from utils._domain import UniformDomain
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import v_relative, clo_dynamic
from pythermalcomfort.utilities import met_typical_tasks
from pythermalcomfort.utilities import clo_individual_garments


def _PMV(tdb, rh, v):
    # input variables
    # tdb \in [12, 30]
    # rh \in [30, 60]
    # v \in [0, 1.5]


    # tdb = 27  # dry bulb air temperature, [$^{\circ}$C]
    tr = tdb - 1 #25 # tdb - 1 # 25  # mean radiant temperature, [$^{\circ}$C]
    # v = 0.3  # average air speed, [m/s]
    # rh = 50  # relative humidity, [%]
    activity = "Typing"  # participant's activity description
    garments = ["Sweatpants", "T-shirt", "Shoes or sandals"]

    met = met_typical_tasks[activity]  # activity met, [met]
    icl = sum([clo_individual_garments[item] for item in garments])  # calculate total clothing insulation

    # calculate the relative air velocity
    vr = v_relative(v=v, met=met)
    # calculate the dynamic clothing insulation
    clo = clo_dynamic(clo=icl, met=met)

    # calculate PMV in accordance with the ASHRAE 55 2020
    results = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, standard="ASHRAE")
    # print(results['ppd'][0])
    # Return the results
    # print(results)
    pmv = results['pmv']
    return pmv

class ThermalComfortTask:
    def __init__(self, dtype=torch.float64):
        self.dtype = dtype
        self.global_minimum = 0
        
    def set_domains(self):
        self.bounds = torch.tensor([
            [12,30,0],
            [30,60,1.5],
        ]).to(self.dtype)
        domain = UniformDomain(self.bounds)
        return domain
    
    def query(self, x):
        x = x.squeeze()
        if len(x.shape) == 2:
            raise ValueError
        
        y = _PMV(x[0], x[1], x[2])
        return torch.tensor([y]).to(self.dtype).abs()
    
    def __call__(self, X):
        X = X.squeeze()
        if len(X.shape) == 2:
            y = torch.cat([self.query(x) for x in X])
        else:
            y = self.query(X)
        return y
        