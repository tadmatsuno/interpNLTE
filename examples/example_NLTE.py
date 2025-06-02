from interpNLTE.Lind22 import corrections
from astropy.table import Table

data = Table(\
    {"teff":[6000,4500],"logg":[4.0,1.5],
     "feh":[-2.5,-2.5],"vt":[1.5,1.5],"EW_5890":[90.,90.]})
corrections.get_corrections(data,["EW_5890"],"Na")