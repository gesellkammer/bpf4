OUTBOUND_DONOTHING = 0
OUTBOUND_CACHE = 1
OUTBOUND_SET = 2

CONFIG = {
    'plot.always_show' : False,
    'preapply.calculate_bounds' : True,
    'crop.outbound_mode' : OUTBOUND_CACHE,
    'integrate.oversample' : 10,
    'integrate.trapz_intervals' : 400,
    'integrate.default_mode':2  # -1: calibrate, 0: trapz, 1: quad, 2: simpsons
}   


