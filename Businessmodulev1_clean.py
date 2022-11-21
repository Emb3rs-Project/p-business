# import json
import re
from typing import Dict

import matplotlib.pyplot as plt
import mpld3
import numpy
import numpy as np
import numpy_financial as npf
import pandas as pd
from jinja2 import Environment, FileSystemLoader  #, PackageLoader, select_autoescape
from pydantic import BaseModel, ValidationError, validator
from .cases.exceptions.module_validation_exception import ModuleValidationException


class TestInput(BaseModel):  # STRUCTURE VALIDATION
# platform inputs
    projectduration: int
    actorshare: list
    discountrate_i: list
    rls: list
    
# MM inputs
    price_h: list
    op_cost: dict
    Pn: dict

# TEO inputs
    opex_values: list
    capex_names: list
    sal_values: list
    capex_values: list
    

# plat form inputs test
    @validator('projectduration')
    def porjectdurationvalidity(cls, v, values, **kwargs):
        if v < 1:
            raise ValueError("Project lifetime cannot be zero. Please provide a valid value for project lifetime.")
        return v

    #@validator('actorshare')
    #def sumofactorshare(cls, v, values, **kwargs):

    #    if sum(v) != 1:
    #        raise ValueError(
    #            "Grid ownership struture is wrongfully definied. The sum of share of all actors should be equal to 1")
    #    return v

    @validator('discountrate_i')
    def discountratevalidity(cls, v, values, **kwargs):
        if min(v) <= 0:
            raise ValueError("Discount rates must be grater than 0")
        return v

    @validator('rls')
    def teahownershipvalidity(cls, v, values, **kwargs):
        if len(v[0]) != 2:
            raise ValueError("Error in Technology ownership. Columns should be 2.")
        #if len(v) != len(values["capex_tt"])+len(values["capex_st"]):
         #   raise ValueError("Error in Technology ownership. Rows must be equal to number of technologies, including storages.")
        return v

## MM input tests
    @validator('Pn') # change this name
    def dispatchvalidity(cls, v, values, **kwargs):
        #vv= list(v.items())
       # vv= gg.astype(float)
        if len(v) != len(values['op_cost']):
            raise ValueError("Dispatch or operating cost missing for one or more actors.")
         #if len(v[0]) != len(values['price_h']):
          #  raise ValueError("Dispatch or price missing for one or more timesteps.")
    #    rlsnp = np.array(values["rls"])
    #    max = rlsnp.max(axis=0, keepdims=True)
    #    mm = max[0]
    #    if mm[0] + 1 != len(values["op_cost"]):
    #        raise ValueError(
    #            "operating cost missing for one or more actors or ownership is not defined for all the actors")
        return v

## TEO input test
    @validator('capex_values')
    def capexvalidity(cls, v, values, **kwargs):
        if len(v) != len(values["opex_values"]):
            raise ValueError("Capex and Opex must be defined for each technology.")
        if len(v) != len(values["capex_names"]):
            raise ValueError("Capex and their names must be defined for each technology.")
        if len(v) != len(values["sal_values"]):
            raise ValueError("Capex and salvage costs must be defined for each technology.")
        return v


### Error handling ends



def BM(input_dict: Dict, generate_template: bool = True) -> Dict:
    # --------------------------------------------------------------------------
    #                         Pre-proccessing / Data preparation START
    # ---------------------------------------------------------------------------

    # input dictionary
    Platform = input_dict["platform"]
    market = input_dict["market-module"]
    teo = input_dict["teo-module"]
    GIS = input_dict["gis-module"]

    # input extraction from dictionaries
    projectduration = Platform["projectduration"]  # int
    actorshare = Platform["actorshare"]  # int
    discountrate_i = np.array(Platform["discountrate_i"])
    # important connects actors (first col) with different tech (second col)
    rls = Platform["rls"]
    rls_map = Platform["rls_map"]
    net_cost = np.array(GIS["net_cost"])

    ### input extraction from MM (dict is market)
    Pn = market["Pn"]
    op_cost = market["agent_operational_cost"]
    price_h = np.array(market["shadow_price"])

    ### input extraction from TEO (dict is teo)
    incoming = teo["DiscountedCapitalInvestmentByTechnology"]
    capex_t_names = [d['TECHNOLOGY'] for d in incoming]
    capex_t_values = [d['VALUE'] for d in incoming]

    incoming = teo["DiscountedCapitalInvestmentByStorage"]
    capex_s_names = [d['STORAGE'] for d in incoming]
    capex_s_values = [d['VALUE'] for d in incoming]

    incoming = teo["DiscountedSalvageValueByTechnology"]
    sal_t_values = [d['VALUE'] for d in incoming]

    incoming = teo["DiscountedSalvageValueByStorage"]
    sal_s_values = [d['VALUE'] for d in incoming]

    incoming = teo["TotalDiscountedFixedOperatingCost"]
    opex_t_values = [d['VALUE'] for d in incoming]

    capex_names = np.concatenate((capex_t_names, capex_s_names))
    capex_values = np.concatenate((capex_t_values, capex_s_values))
    sal_values = np.concatenate((sal_t_values, sal_s_values))
    opex_values = np.concatenate((opex_t_values, [0] * len(capex_s_names)))

    ####################################################3


# Input Error check - Error handling ---- Starts---

    Pnn = np.array(market["Pn"])
    mm = {
        "price_h": price_h.tolist(),
        "Pn" : Pnn.tolist(),
        "op_cost" : op_cost
    }
    teo={
        "capex_names":capex_names.tolist(),
        "capex_values":capex_values.tolist(),
        "sal_values":sal_values.tolist(),
        "opex_values":opex_values.tolist()
    }
    _indict = { **Platform, **mm, **teo}
    try:
        _model = TestInput(**_indict)

    # print(_model.schema_json(indent=2))

    except ValidationError as e:
        raise ModuleValidationException(code=1, msg="Problem with Business module", error=e)
    #except Exception as e:
#    print(e)

# Input Error check - Error handling ---- Ends---

####################################################3

####################################################3

    mm_agents = []
    dispatch_ih = np.zeros(shape=(len(Pn), 8760))
    op_cost_i = []
    count = 0
    for i in Pn.keys():
        dispatch_ih[count] = Pn[i]

        opp = np.array(op_cost[i])
        op_cost_i.append(sum(opp))

        mm_agents.append(i)
        count = count + 1

    ## combining everything at source and sink resolution for MM

    mm_agents_copy = mm_agents
    mm_agents_fil = []
    for i in mm_agents_copy:
        if i.find("grid") >= 0:
            tech = "grid"
        elif i.find("sou") >= 0:
            dig = re.findall("sou\B([0-9]+)", i)
            tech = "source" + dig[0]
        elif i.find("sink") >= 0:
            dig = re.findall("sink\B([0-9]+)", i)
            tech = "sink" + dig[0]
        elif i.find("dhn") >= 0:
            tech = "dhn"

        mm_agents_fil.append(tech)

    total_dispatch_mi = np.sum(dispatch_ih, axis=1)
    revenues_mi = abs(dispatch_ih * price_h)
    total_revenues_mi = np.sum(revenues_mi, axis=1)

    data_df_m = {
        "agent_m": mm_agents_fil,
        "total_dispatch": total_dispatch_mi.tolist(),
        "total_rev": total_revenues_mi.tolist(),
        "op_cost_i": op_cost_i
    }

    mdf = pd.DataFrame(data_df_m)
    mdf = mdf[(mdf.agent_m != "dhn")]
    mdf_uniq = mdf.groupby('agent_m').sum()

    ## combining everything at source and sink resolution for TEO

    teo_agents_copy = capex_names
    teo_agents_fil = []
    for i in teo_agents_copy:
        if i.find("grid") >= 0:
            tech = "grid"
        elif i.find("storage") >= 0:
            tech = "grid"
        elif i.find("sou") >= 0:
            dig = re.findall("sou\B([0-9]+)", i)
            tech = "source" + dig[0]
        elif i.find("sink") >= 0:
            dig = re.findall("sink\B([0-9]+)", i)
            tech = "sink" + dig[0]

        teo_agents_fil.append(tech)

    data_df_teo = {
        "agent_teo": teo_agents_fil,
        "capex_values": capex_values,
        "sal_values": sal_values,
        "opex": opex_values
    }

    teodf = pd.DataFrame(data_df_teo)
    teodf_uniq = teodf.groupby('agent_teo').sum()

    # taking out grid part
    #mdf_uniq = mdf_uniq.iloc[1:]

    #grid = teodf_uniq.iloc[0]
    #teodf_uniq = teodf_uniq.iloc[1:]

    ######### rls operation
    mdf_uniq["owner"] = " "
    teodf_uniq["owner"] = " "
    mdf_uniq['agent_market'] = mdf_uniq.index
    teodf_uniq['agent_techno'] = teodf_uniq.index
    mdf_uniq = mdf_uniq.drop(mdf_uniq[mdf_uniq.agent_market.str.contains("grid")].index)
    mdf_uniq = mdf_uniq.drop(mdf_uniq[mdf_uniq.agent_market.str.contains("dhn")].index)
    grid_idx = teodf_uniq[teodf_uniq.agent_techno.str.contains("grid")].index
    grid = teodf_uniq.loc["grid"]
    teodf_uniq = teodf_uniq.drop(teodf_uniq[teodf_uniq.agent_techno.str.contains("grid")].index)
    teodf_uniq = teodf_uniq.drop(teodf_uniq[teodf_uniq.agent_techno.str.contains("dhn")].index)

    for i in rls:
        actor = i[0].replace(" ", "")
        tech_temp = i[1]
        tt = tech_temp.split()
        tech_real = tt[0] + tt[1]
        mdf_uniq.loc[mdf_uniq.index[mdf_uniq["agent_market"] == tech_real], "owner"] = actor
        teodf_uniq.loc[teodf_uniq.index[teodf_uniq["agent_techno"] == tech_real], "owner"] = actor

    mdf_uniq = mdf_uniq.reset_index(drop=True)
    teodf_uniq = teodf_uniq.reset_index(drop=True)
    lis1 = mdf_uniq["owner"].tolist()
    lis2 = mdf_uniq["agent_market"].tolist()
    lis = list(set(lis2).difference(lis1))
    for i in lis:
        mdf_uniq.loc[len(mdf_uniq), "owner"] = i
        teodf_uniq.loc[len(teodf_uniq), "owner"] = i

    mdf_uniq = mdf_uniq.fillna(0)
    teodf_uniq = teodf_uniq.fillna(0)

    mdf_uniq = mdf_uniq.groupby("owner").sum()
    teodf_uniq = teodf_uniq.groupby("owner").sum()

    mdf_uniq['owner2'] = mdf_uniq.index
    teodf_uniq['owner2'] = teodf_uniq.index

    ##### getting final parameters

    capex_i = teodf_uniq["capex_values"].to_numpy()
    sal_i = teodf_uniq["sal_values"].to_numpy()
    capex_i = capex_i - sal_i
    opex_i = teodf_uniq["opex"].to_numpy()
    actors_i = teodf_uniq.index

    s = []
    s_names = []
    actor_names = []
    count = 0
    for i in actors_i:

        if i.find("sink") >= 0:
            s.append(count)
            s_names.append(i)
        else:
            actor_names.append(i)
        count = count + 1

    real_actor_names = actor_names.copy()
    real_s_names = s_names.copy()

    count = 0
    for i in actor_names:
        name = " "
        for j in rls_map:
            if i == j[0].replace(" ", ""):
                name = j[1]

        real_actor_names[count] = name
        count = count + 1
    count = 0
    for i in s_names:
        name = " "
        for j in rls_map:
            if i == j[0].replace(" ", ""):
                name = j[1]

        real_s_names[count] = name
        count = count + 1

    s_names = real_s_names
    actor_names = real_actor_names

    dispatch_i = mdf_uniq["total_dispatch"].to_numpy()
    revenues_i = mdf_uniq["total_rev"].to_numpy()
    opcost_i = mdf_uniq["op_cost_i"].to_numpy()

    # Adding share of network cost to each actors capex
    capex_i = capex_i + ( (net_cost + grid["capex_values"] - grid["sal_values"])*np.array(actorshare))
    # seperating sink
    capex_s = capex_i[s]
    opex_s = opex_i[s]
    opcost_s = opcost_i[s]
    heat_cost_s = revenues_i[s]  # money spent by sink to buy heat
    dispatch_s = (-1) * dispatch_i[
        s
    ]  # make sure dispatch also take into account energy consumed by sink
    # seperating sources
    capex_i = np.delete(capex_i, s)
    opex_i = np.delete(opex_i, s)
    opcost_i = np.delete(opcost_i, s)
    revenues_i = np.delete(revenues_i, s)

    # --------------------------------------------------------------------------
    #                         Pre-proccessing / Data preparation END
    # ---------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    #                         business mode conditional statement START
    # ---------------------------------------------------------------------------
    # if socio-economic (maybe not coz gonna calculate both scenario)
    capex = np.sum(capex_i) + np.sum(capex_s)
    opex = np.sum(opex_i) + np.sum(opex_s) + grid["opex"]
    revenues = np.sum(revenues_i) - np.sum(heat_cost_s)
    op_cost = np.sum(opcost_i) + np.sum(opcost_s)
    r = discountrate_i[0]

    # +- 50% variations with total 5 values in R
    r_sen = np.linspace(r * 0.5, r * 1.5, 5)
    y = projectduration

    # NPV calculation
    netyearlyflow = revenues - op_cost
    if netyearlyflow.size == 0:
        netyearlyflow = 0
    sumyearlyflow = 0
    rev_yearly = 0
    op_cost_yearly = 0
    for i in range(1, y + 1):
        sumyearlyflow += netyearlyflow / (1 + r) ** i
        rev_yearly += revenues / (1 + r) ** i
        op_cost_yearly += op_cost / (1 + r) ** i
    NPV_socio = sumyearlyflow - capex - opex
    if rev_yearly == 0:
        rev_yearly = 0.000000001
    PayBack_socio = (capex + opex + op_cost_yearly) / (rev_yearly)
    sumyearlyflow = 0
    for i in range(1, y + 1):
        sumyearlyflow += netyearlyflow / (1 + r_sen) ** i

    NPV_socio_sen = sumyearlyflow - capex - opex

    # np.append(-capex, np.full(y, netyearlyflow))
    IRR_socio = npf.irr(np.append(-(capex + opex), np.full(y, netyearlyflow)))

    # ------------------------------------------
    # if Business
    # -----------------------------------------

    # NPV & IRR calculation

    r_b = discountrate_i[1]
    # +- 50% variations with total 5 values in R
    r_sen_b = np.linspace(r_b * 0.5, r_b * 1.5, 5)

    # 1D array[ actor1, actor2, ...., actorX]
    netyearlyflow_i = revenues_i - opcost_i - opex_i / y
    if netyearlyflow_i.size == 0:
        netyearlyflow_i = 0
    sumyearlyflow_i = np.zeros(len(netyearlyflow_i))
    for i in range(1, y + 1):
        sumyearlyflow_i += netyearlyflow_i / (1 + r_b) ** i  # 1D array

    NPV_i = sumyearlyflow_i - capex_i

    IRR_i = np.zeros(netyearlyflow_i.size)

    for i in range(0, netyearlyflow_i.size):
        IRR_i[i] = npf.irr(np.append(-capex_i[i], np.full(y, netyearlyflow_i[i])))

    # Payback period for each actor

    rev_new = revenues_i
    for i in range(0, rev_new.size):
        if rev_new[i] == 0:
            rev_new[i] = float("nan")

    rev_yearly = 0
    op_cost_yearly = 0
    for i in range(1, y + 1):
        rev_yearly += rev_new / (1 + r_b) ** i
        op_cost_yearly += np.array(opcost_i) / (1 + r_b) ** i

    PayBack_i = (capex_i + opex_i + op_cost_yearly) / (rev_yearly)


    # sensitivity ananlysis for NPV

    # from 1D array to row matrix[ [actor1], [actor2], ...., [actorX]]
    netyearlyflow_i = netyearlyflow_i[np.newaxis]

    sumyearlyflow_i = np.zeros(netyearlyflow_i.size).reshape((-1, 1))  # column matrix
    for j in range(0, r_sen_b.size):
        sumyearlyflow_i_temp = 0
        for i in range(1, y + 1):
            # row matrix (rows represents cash flow for each actor at a given r (column))
            sumyearlyflow_i_temp += netyearlyflow_i / (1 + r_sen_b[j]) ** i

        # row to column matrix
        temp = np.reshape(sumyearlyflow_i_temp, (netyearlyflow_i.size, 1))
        # 2D matric with rows for each actor and column for different r
        sumyearlyflow_i = np.append(sumyearlyflow_i, temp, axis=1)

    # removing first col as it is zero
    sumyearlyflow_i = np.delete(sumyearlyflow_i, 0, 1)
    capex_i = capex_i.reshape((-1, 1))  # row to column matrix

    NPV_sen_i = sumyearlyflow_i - capex_i  # 2D matrix

    # >>>>>>>>> LCOH calculation
    LCOH_sen = np.zeros((opex_s.size, 1))  # column matrix
    for j in range(0, r_sen_b.size):
        sumrevflow = 0
        sumdisflow = 0
        for i in range(1, y + 1):
            sumrevflow += (opex_s / y + opcost_s + heat_cost_s) / (1 + r_sen_b[j]) ** i
            sumdisflow += dispatch_s / (1 + r_sen_b[j]) ** i
        for k in range(0, sumdisflow.size):
            if sumdisflow[k] == 0:
                sumdisflow[k] = float("nan")
        LCOH_s = (capex_s + sumrevflow) / sumdisflow
        LCOH_s = LCOH_s.reshape((-1, 1))
        LCOH_sen = np.append(LCOH_sen, LCOH_s, axis=1)

    LCOH_sen = np.delete(LCOH_sen, 0, 1)

    # ----------------------------------------------------------------
#                     Polots & Reporting
#----------------------------------------------------------------
    fig1, ax = plt.subplots()
    ax.plot(r_sen, NPV_socio_sen)
    plt.ylabel('NPV')
    plt.xlabel('Discount rate')
    #plt.title('NPV for Socio-economic Scenario')

    fig2, ax = plt.subplots()
    for i in range(0, netyearlyflow_i.size):
        ax.plot(r_sen_b, NPV_sen_i[i, :], label="%s" % actor_names[i])
    plt.legend(loc="upper left")
    plt.ylabel('NPV')
    plt.xlabel('Discount rate')
    #plt.title('NPV for Business Scenario')

    fig3, ax = plt.subplots()
    for i in range(0, opex_s.size):
        ax.plot(r_sen_b, LCOH_sen[i, :], label="%s" % s_names[i])
    plt.legend(loc="upper left")
    plt.ylabel('LCOH - €/kWh')
    plt.xlabel('Discount rate')
    #plt.title('LCOH for Sinks')

    fig1ht = mpld3.fig_to_html(fig1)
    fig2ht = mpld3.fig_to_html(fig2)
    fig3ht = mpld3.fig_to_html(fig3)

    import os
    script_dir = os.path.dirname(__file__)

    env = Environment(
        loader=FileSystemLoader(os.path.join(script_dir, "asset")),
        autoescape=False
    )

    template = env.get_template('BMtemplatev1.html')
    template_content = template.render(plotNPVs=fig1ht, IRR_socio=np.around(IRR_socio, decimals=4), plotNPVb=fig2ht,
                                       plotLCOH=fig3ht, actor_names=actor_names, IRR_i=np.around(IRR_i, decimals=4),
                                       Payback_socio=np.around(PayBack_socio, decimals=4),
                                       Payback_actors=np.around(PayBack_i, decimals=4))

    if generate_template:
        f = open("index.html", "w")
        f.write(template_content)
        f.close()
    #plt.show()

    output = {
        "NPV_socio-economic": NPV_socio.tolist(),
        "IRR_socio-economic": IRR_socio,
        "Sensitivity_NPV_socio-economic": NPV_socio_sen.tolist(),
        "NPV_comm_actor": NPV_i.tolist(),
        "IRR_comm_actor": IRR_i.tolist(),
        "Sensitivity_NPV_comm_actor": NPV_sen_i.tolist(),
        "Discountrate_socio": r_sen.tolist(),
        "Discountrate_business": r_sen_b.tolist(),
        "LCOH_s": LCOH_s.tolist(),
        "Payback_socio": PayBack_socio.tolist(),
        "Payback_actors": PayBack_i.tolist(),
        "report": template_content,
    }
    return output


def int_heat_rec(heat_rec_input_dict):
    CF = heat_rec_input_dict["cf-module"]

    c = CF["capex"]
    of = CF["O&M_fix"]
    d = CF["energy_dispatch"]
    r = CF["discount_rate"]
    rev = CF["money_sav"]
    c_q = CF["carbon_sav_quant"]
    #   c_m = heat_rec_input_dict["carbon_sav_money"]
    n = CF["duration"]

    r_sen = np.linspace(r * 0.5, r * 1.5, 5)
    # >>>>>>>>> LCOH calculation
    sumrevflow = 0
    sumdisflow = 0
    for i in range(1, n + 1):
        sumrevflow += of / (1 + r_sen) ** i
        sumdisflow += d / (1 + r_sen) ** i

    LCOH = (c + sumrevflow) / sumdisflow

    # >>>>>>>>> NPV calculation

    netyearlyflow = rev - of  # c_m if provided would be used here

    sumyearlyflow = 0
    for i in range(1, n + 1):
        sumyearlyflow += netyearlyflow / (1 + r_sen) ** i

    NPV = sumyearlyflow - c


#---------------------------------------------------------
#       Plotting and reporting
#------------------------------------------------------

    fig1, ax = plt.subplots()
    ax.plot(r_sen, NPV)
    plt.ylabel('NPV')
    plt.xlabel('Discount rate')

    fig2, ax = plt.subplots()
    ax.plot(r_sen, LCOH)
    plt.ylabel('LCOH - €/kWh')
    plt.xlabel('Discount rate')

    fig1ht = mpld3.fig_to_html(fig1)
    fig2ht = mpld3.fig_to_html(fig2)

    env = Environment(
        loader=FileSystemLoader('asset'),
        autoescape=False
    )

    template = env.get_template('intheattemplatev1.html')
    template_content = template.render(plotNPVs=fig1ht, plotLCOH=fig2ht)

    f = open("index2.html", "w")
    f.write(template_content)
    f.close()



    # >>> Output dictionary

    heat_rec_output_dict = {"LCOH_sen": LCOH.tolist(), "NPV_sen": NPV.tolist(), "report": template_content}

    return heat_rec_output_dict
