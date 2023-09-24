# Author: Seprodi Barnabas
# Version: 1
# Date: 2023-07-02

import matplotlib.pyplot as plt
import numpy as np
from ArresterBedRadar import *

def main():
    print("Converter tools to convert old csv files")
    filename_old = require("Old filename: ",str)
    X,Y,Z = importCsv(f"measurements/{filename_old}.csv")
    Z = -Z

    for y in np.unique(Y):
        ind = np.where(Y == y)
        x = X[ind]
        z = Z[ind]
        w,h,A = stats(x,z)

        with open(f"measurements/{filename_old}-new-stats.csv","a+") as f:
            f.write(f"{y}; {w}; {h}; {A}\n".replace(".",","))
        f.close()

        with open(f"measurements/{filename_old}-new.csv","a+") as f:
            for i in range(len(x)):
                f.write(f"{x[i]}; {y}; {z[i]}\n".replace(".",","))
        f.close()
    print(f"Points saved to 'measurements/{filename_old}-new.csv'")
    print(f"Statistics saved to 'measurements/{filename_old}-new-stats.csv'")

if __name__ == "__main__":
    main()
