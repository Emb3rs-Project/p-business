import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt


def BM(BM_input_dict):

    # --------------------------------------------------------------------------
    #                         Pre-proccessing / Data preparation START
    # ---------------------------------------------------------------------------
    # input dictionary

    Platform = BM_input_dict["platform"]
    MM = BM_input_dict["market-module"]
    TEO = BM_input_dict["teo-module"]

    GIS = BM_input_dict["gis-module"]

    # input extraction from dictionary

    # MWh (per actor per hour for a whole year), int, 2D array
    dispatch_ih = np.array(MM["dispatch_ih"])
    # EUR/MWh (per actor per hour for a whole year) float, 1D array
    price_h = np.array(MM["price_h"])
    # EUR (per hour per actor for a whole year), float, 2D array
    opcost_i = np.array(MM["opcost_i"])
    # OR if no op_cost is given; then
    capex_tt = np.array(TEO["capex_tt"])  # EUR, int, 1D array
    # EUR/year, int, 1D array
    opex_tt = np.array(TEO["opex_tt"])
    projectduration = Platform["projectduration"]  # int
    actorshare = Platform["actorshare"]  # int
    discountrate_i = np.array(Platform["discountrate_i"])
    # important connects actors (first col) with different tech (second col)
    rls = np.array(Platform["rls"], dtype=int)
    s = np.array(Platform["sinks"], dtype=int)
    capex_st = np.array(TEO["capex_st"])
    capex_t_names = np.array(TEO["capex_t_names"])
    capex_s_names = np.array(TEO["capex_s_names"])
    sal_tt = np.array(TEO["sal_tt"])
    sal_st = np.array(TEO["sal_st"])
    net_cost = np.array(GIS["net_cost"])

    # ----------------------------
    # Error Check on inputs START
    # -------------------------------

    actors = opcost_i.size
    row, col = dispatch_ih.shape
    hours_yr = col

    if discountrate_i[0] == 0 or discountrate_i[1] == 0:
        raise Exception("Discount rate cannot be zero. Please provide valid values for discount rates. ")
    if projectduration < 1:
        raise Exception("Project lifetime cannot be zero. Please provide a valid value for project lifetime.")
    row, col = dispatch_ih.shape
    if row != actors:
        raise Exception("Heat dispatch missing for one or more actors. Please provide heat dispatch for each actor.")
    if col != hours_yr:
        raise Exception(
            "Heat dispatch missing for one or more hours. Please provide heat dispatch for each hour in a year.")
    # this one moved to revenues
    # print(price_h)
    # row, col = price_h.shape
    # if col != hours_yr:
    #    raise Exception("Price missing for one or more hours. Please provide heat price for each hour in a year.")
    if opcost_i.size != actors:
        raise Exception("Operational cost missing for one or more actors.")
    if capex_tt.size != opex_tt.size:
        raise Exception("The capital costs or O&M cost missing for one or more technology.")
    row, col = rls.shape
    if row != capex_tt.size + capex_st.size:
        raise Exception(
            "The ownership structure is not defined correctly. All the technologies must belong to an actor.")
    if s.size < 1:
        raise Exception("There are no sinks in the simulation.")
    if sal_tt.size != capex_tt.size:
        raise Exception("The salvage costs are not defined for one or more technologies :: TEO input error.")
    if sal_st.size != capex_st.size:
        raise Exception("The salvage costs are not defined for one or more storage units:: TEO input error.")

    # ----------------------------
    # Error Check on inputs ENDS
    # -------------------------------

    # combining capex and salvage cost for tech and storages
    capex_t = np.concatenate((capex_tt, capex_st))
    sal_t = np.concatenate((sal_tt, sal_st))
    opex_t = np.pad(opex_tt, (0, np.size(capex_st)), "constant")

    # capex & opex from tech to actors

    i = np.copy(rls)
    temp = capex_t[i[:, 1]]
    i[:, 1] = temp
    temp_i = np.zeros((np.max(i, axis=0)[0] + 1, np.shape(i)[1]))
    temp_i[0 : np.max(i, axis=0)[0] + 1, 0] = [
        np.sum(i[i[:, 0] == j, 1]) for j in range(np.max(i, axis=0)[0] + 1)
    ]
    capex_i = temp_i[0 : np.max(i, axis=0)[0] + 1][:, 0]

    i = np.copy(rls)
    temp = sal_t[i[:, 1]]
    i[:, 1] = temp
    temp_i = np.zeros((np.max(i, axis=0)[0] + 1, np.shape(i)[1]))
    temp_i[0 : np.max(i, axis=0)[0] + 1, 0] = [
        np.sum(i[i[:, 0] == j, 1]) for j in range(np.max(i, axis=0)[0] + 1)
    ]
    sal_i = temp_i[0 : np.max(i, axis=0)[0] + 1][:, 0]
    capex_i = capex_i - sal_i

    i = np.copy(rls)
    temp = opex_t[i[:, 1]]
    i[:, 1] = temp
    temp_i = np.zeros((np.max(i, axis=0)[0] + 1, np.shape(i)[1]))
    temp_i[0 : np.max(i, axis=0)[0] + 1, 0] = [
        np.sum(i[i[:, 0] == j, 1]) for j in range(np.max(i, axis=0)[0] + 1)
    ]
    opex_i = temp_i[0 : np.max(i, axis=0)[0] + 1][:, 0]

    # Operational cost
    # opcost_ih = co2taxtot_ih + fuelcost_ih #optional
    # total operational cost of actors in a year; 1D array
    # opcost_i = np.sum(opcost_ih, axis=1)

    # Revenues
    revenues_ih = abs(dispatch_ih * price_h)

    # ----------------------------
    # Error Check on revinues starts
    # -------------------------------
    row, col = revenues_ih.shape
    if col != hours_yr:
        raise Exception("Price missing for one or more hours. The hourly price must be provided for each hour.")

    # ----------------------------
    # Error Check on revinues ends
    # -------------------------------


    # total revenues of actors in a year; 1D array
    revenues_i = np.sum(revenues_ih, axis=1)
    dispatch_i = np.sum(dispatch_ih, axis=1)
    # Adding share of network cost to each actors capex
    capex_i = capex_i + (net_cost*actorshare)
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
    capex = np.sum(capex_i) + np.sum(capex_s) + net_cost
    opex = np.sum(opex_i) + np.sum(opex_s)
    revenues = np.sum(revenues_i) + np.sum(heat_cost_s)
    op_cost = np.sum(opcost_i) + np.sum(opcost_s)
    r = discountrate_i[0]

    # +- 50% variations with total 5 values in R
    r_sen = np.linspace(r * 0.5, r * 1.5, 5)
    y = projectduration

    # NPV calculation
    netyearlyflow = revenues - op_cost

    sumyearlyflow = 0
    for i in range(1, y + 1):
        sumyearlyflow += netyearlyflow / (1 + r) ** i

    NPV_socio = sumyearlyflow - capex - opex

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
    sumyearlyflow_i = np.zeros(len(netyearlyflow_i))
    for i in range(1, y + 1):
        sumyearlyflow_i += netyearlyflow_i / (1 + r_b) ** i  # 1D array

    NPV_i = sumyearlyflow_i - capex_i

    IRR_i = np.zeros(netyearlyflow_i.size)

    for i in range(0, netyearlyflow_i.size):
        IRR_i[i] = npf.irr(np.append(-capex_i[i], np.full(y, netyearlyflow_i[i])))

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
    sumrevflow = 0
    sumdisflow = 0
    for i in range(1, y + 1):
        sumrevflow += (opex_s / y + opcost_s + heat_cost_s) / (1 + r_b) ** i
        sumdisflow += dispatch_s / (1 + r_b) ** i

    LCOH_s = (capex_s + sumrevflow) / sumdisflow

    BM_output = {
        "NPV_socio-economic": NPV_socio.tolist(),
        "IRR_socio-economic": IRR_socio,
        "Sensitivity_NPV_socio-economic": NPV_socio_sen.tolist(),
        "NPV_comm_actor": NPV_i.tolist(),
        "IRR_comm_actor": IRR_i.tolist(),
        "Sensitivity_NPV_comm_actor": NPV_sen_i.tolist(),
        "Discountrate_socio": r_sen.tolist(),
        "Discountrate_business": r_sen_b.tolist(),
        "LCOH_s": LCOH_s.tolist(),
    }

    return BM_output


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

    # >>> Output dictionary

    heat_rec_output_dict = {"LCOH_sen": LCOH.tolist(), "NPV_sen": NPV.tolist()}

    return heat_rec_output_dict
