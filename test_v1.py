import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
from Businessmodulev1_clean import BM
# --------------------------------------------------------------------------
#                         INPUTS START
#Note: subscript i correspond to actors, h hours, t to each technology component.
# correspondence of i to t should be mapped i.e i = 0 ; t= 0,1,4
#---------------------------------------------------------------------------

#           %From MM%

dispatch_ih = np.array([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50], [20, 30, 40, 50, 60]], dtype=int)   # MWh (per actor per hour for a whole year), int, 2D array
price_h=   np.array([1, 2, 3, 4, 5])     # EUR/MWh (per actor per hour for a whole year) float, 1D array
opcost_ih = np.array([[0, 0, 0, 0, 0], [1, 2, 3, 4, 5], [2, 30, 40, 5, 60]])   # EUR (per hour per actor for a whole year), float, 2D array
# OR if no op_cost is given; then
#co2taxtot_ih =  #EUR (per hour per actor for a whole year), float, 2D array
#fuelcost_ih = #EUR (per hour per actor for a whole year), float, 2D array

#           %From TEO%

capex_t  = np.array([100, 200, 300, 400, 500, 600, 700], dtype=int) # EUR, int, 1D array
opex_t   = np.array([10, 20, 30, 40, 50, 60, 70], dtype=int)# EUR/year, int, 1D array

#           %From KB%

projectduration =  5   # int
discountrate_i =   np.array([3, 5.5])   # float, 1D array with two elements (first socio-economic discount rate, second private one - simplified right now, doesn't take into account different discount rates per tech and/or actor)
#emissionfactortech_t =   # float CO2e Kg/MWh, 1D array
#heatsupplyco2ent =    # floeat CO2e Kg/MWh

#       %From User / Platform%

#businessmode =        # Binary 1= socio; 0= business

# ownership structure
rls = np.array([[0, 0], [0, 2], [1, 1], [1, 3], [1, 6], [2, 4], [2, 5]], dtype=int) ### important connects acors (first col) with different techs (second col)


# --------------------------------------------------------------------------
#                         INPUTS END
#---------------------------------------------------------------------------

# --------------------------------------------------------------------------
#                         FUNCTION CALL START
#---------------------------------------------------------------------------

NPV_socio, IRR_socio, NPV_socio_sen, NPV_i, IRR_i, NPV_sen_i, r_sen, r_sen_b = BM(rls, capex_t, opex_t, opcost_ih, dispatch_ih, price_h, discountrate_i, projectduration)

### OUTPUT AND JSON CONVERSION

BM_output = {
    'NPV_socio-economic' : NPV_socio,
    'IRR_socio-economic' : IRR_socio,
    'Sensitivity_NPV_socio-economic' : NPV_socio_sen.tolist(),
    'NPV_comm_actor' : NPV_i.tolist(),
    'IRR_comm_actor' : IRR_i.tolist(),
    'Sensitivity_NPV_comm_actor'  : NPV_sen_i.tolist(),
    'Discountrate_socio' : r_sen.tolist(),
    'Discountrate_business' : r_sen_b.tolist()
}

BM_output_json = js.dumps(BM_output)


# --------------------------------------------------------------------------
#                         FUNCTION CALL END
#---------------------------------------------------------------------------

