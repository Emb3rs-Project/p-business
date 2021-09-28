import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt

# function arrguements

def BM(rls, capex_t, opex_t, opcost_ih, dispatch_ih, price_h, discountrate_i, projectduration):

# --------------------------------------------------------------------------
#                         Pre-proccessing / Data preparation START
#---------------------------------------------------------------------------

#capex & opex from tech to actors

    i= np.copy(rls)
    temp = capex_t[i[:, 1]]
    i[:, 1] = temp
    temp_i = np.zeros((np.max(i, axis=0)[0] + 1, np.shape(i)[1]))
    temp_i[0:np.max(i, axis=0)[0] + 1, 0] = [np.sum(i[i[:, 0] == j, 1]) for j in range(np.max(i, axis=0)[0] + 1)]
    capex_i = temp_i[0:np.max(i, axis=0)[0] + 1][:, 0]


    i= np.copy(rls)
    temp = opex_t[i[:, 1]]
    i[:, 1] = temp
    temp_i = np.zeros((np.max(i, axis=0)[0] + 1, np.shape(i)[1]))
    temp_i[0:np.max(i, axis=0)[0] + 1, 0] = [np.sum(i[i[:, 0] == j, 1]) for j in range(np.max(i, axis=0)[0] + 1)]
    opex_i = temp_i[0:np.max(i, axis=0)[0] + 1][:, 0]

    # Operational cost
    #opcost_ih = co2taxtot_ih + fuelcost_ih #optional
    opcost_i = np.sum(opcost_ih, axis=1) # total operational cost of actors in a year; 1D array


    # Revenues
    revenues_ih = dispatch_ih * price_h
    revenues_i = np.sum(revenues_ih, axis = 1) # total revenues of actors in a year; 1D array


    # --------------------------------------------------------------------------
    #                         Pre-proccessing / Data preparation END
    #---------------------------------------------------------------------------



    # --------------------------------------------------------------------------
    #                         business mode conditional statement START
    #---------------------------------------------------------------------------
    ###             if socio-economic (maybe not coz gonna calculate both scenario)
    capex = np.sum(capex_i)
    opex = np.sum(opex_i)
    revenues = np.sum(revenues_i)
    op_cost = np.sum(opcost_i)
    r= discountrate_i[0]

    r_sen = np.linspace(r*0.5, r*1.5, 5) # +- 50% variations with total 5 values in R
    y= projectduration

    ### NPV calculation
    netyearlyflow = revenues - op_cost - opex

    sumyearlyflow=0
    for i in range(1, y+1):
        sumyearlyflow += (netyearlyflow/(1 + r)**i)

    NPV_socio = sumyearlyflow - capex

    sumyearlyflow=0
    for i in range(1, y+1):
        sumyearlyflow += (netyearlyflow/(1 + r_sen)**i)

    NPV_socio_sen = sumyearlyflow - capex

    #np.append(-capex, np.full(y, netyearlyflow))
    IRR_socio = npf.irr(np.append(-capex, np.full(y, netyearlyflow)))


    #------------------------------------------
    ###                if Business
    #-----------------------------------------


    ### NPV & IRR calculation

    r_b = discountrate_i[1]
    r_sen_b = np.linspace(r_b * 0.5, r_b * 1.5, 5)  # +- 50% variations with total 5 values in R

    netyearlyflow_i = revenues_i - opcost_i - opex_i  # 1D array[ actor1, actor2, ...., actorX]
    sumyearlyflow_i = np.zeros(len(netyearlyflow_i))
    for i in range(1, y + 1):
        sumyearlyflow_i += (netyearlyflow_i / (1 + r_b) ** i)  # 1D array

    NPV_i = sumyearlyflow_i - capex_i

    IRR_i = np.zeros(netyearlyflow_i.size)

    for i in range(0, netyearlyflow_i.size):
        IRR_i[i] = npf.irr(np.append(-capex_i[i], np.full(y, netyearlyflow_i[i])))

    ### sensitivity ananlysis for NPV

    netyearlyflow_i = netyearlyflow_i[np.newaxis]   # from 1D array to row matrix[ actor1, actor2, ...., actorX]

    sumyearlyflow_i=np.zeros(netyearlyflow_i.size).reshape((-1,1)) # column matrix
    for j in range (0, r_sen_b.size):
       sumyearlyflow_i_temp = 0
       for i in range(1, y+1):
            sumyearlyflow_i_temp += (netyearlyflow_i/(1 + r_sen_b[j])**i) # row matrix (rows represents cash flow for each actor at a given r (column))

       temp = np.reshape(sumyearlyflow_i_temp, (netyearlyflow_i.size, 1)) # row to column matrix
       sumyearlyflow_i = np.append(sumyearlyflow_i, temp, axis=1) # 2D matric with rows for each actor and column for different r

    sumyearlyflow_i = np.delete(sumyearlyflow_i, 0, 1) # removing first col as it is zero
    capex_i = capex_i.reshape((-1,1)) # row to column matrix

    NPV_sen_i = sumyearlyflow_i - capex_i # 2D matrix

# --------------------------------------------------------------------------
#                         Plot function START
#---------------------------------------------------------------------------
    print('IRR for two scenarios')
    print('IRR socio-economic scenario', IRR_socio)
    print('IRR business scenario', IRR_i)

    plt.figure()
    plt.subplot(211)
    plt.plot(r_sen, NPV_socio_sen)
    plt.ylabel('NPV')
    plt.xlabel('Discount rate')
    plt.title('NPV for Socio-economic Scenario')

    plt.subplot(212)
    for i in range(0, netyearlyflow_i.size):
     plt.plot(r_sen_b, NPV_sen_i[i,:], label = "actor %s" % i)

    plt.legend(loc="upper left")
    plt.ylabel('NPV')
    plt.xlabel('Discount rate')
    plt.title('NPV for Business Scenario')

    plt.show()
# --------------------------------------------------------------------------
#                         Plot function END
#---------------------------------------------------------------------------
    return NPV_socio, IRR_socio, NPV_socio_sen, NPV_i, IRR_i, NPV_sen_i