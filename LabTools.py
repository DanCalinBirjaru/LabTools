from asyncio.windows_events import NULL
import numpy as np
import matplotlib.pyplot as plt
import uncertainties as uc
import uncertainties.umath as um
from uncertainties import unumpy as unp

class FileWork():
    def GetMagnitudeWithError(fileName):
        magWithError = np.dtype([("mag", "f8"), ("error", "f8")])

        mag, error = np.loadtxt(fileName, dtype = magWithError, usecols = (3, 4), unpack = True)

        return unp.uarray([mag, error])

    def OffsetPositionFromGaia(inputFile, outputFile, offsetX, offsetY):
        with open(inputFile) as f:
            lines = f.readlines()

        f = open(outputFile, "w")

        for i in range(0, len(lines), 2):
            stringsArray = lines[i].split()

            numberX = float(stringsArray[1])
            numberX += offsetX 
            stringsArray[1] = "{:.6f}".format(numberX)

            numberY = float(stringsArray[2])
            numberY += offsetY 
            stringsArray[2] = "{:.6f}".format(numberY)

            lines[i] = stringsArray
            lines[i].extend(" ")
            lines[i].extend(lines[i + 1].split())
            
            for str in lines[i]:
                f.write(str + " ")
            f.write("\n")

        f.close()

class PlottingTools:
    def LinFit(xData : np.dtype(float), yData : np.dtype(float)):
        '''
        A function that computes values for the intercept and gradient (with errors)
        of a best fit line starting from a given data set.

        This function uses the least squares method.
        '''

        #Getting the length of the data set
        n = len(xData)

        #Getting the average of the xData
        xBar = np.mean(xData)

        #Extra term for computing errors' values
        D = sum(xData**2) - 1./n * sum(xData)**2

        #pCoeff stores the coefficient of the 1st order polynomial describing the data set
        #residuals are used in the error calculation
        pCoeff, residuals, _, _, _ = np.polyfit(xData, yData, 1, full=True)

        #Computing gradient and error
        m = pCoeff[0]
        dmSquared = 1./(n - 2) * residuals / D
        dm = np.sqrt(dmSquared)

        #Computing intercept and error
        c = pCoeff[1]
        dcSquared = 1./(n - 2)*(D / n + xBar**2) * residuals / D
        dc = np.sqrt(dcSquared)

        return m, dm, c, dc

    def Linear(x, *p):
        return p[0] * x + p[1]

    def Quadratic(x, *p):
        return p[0]*x**2 + p[1]*x + p[2]

    def Sinusoidal(x, t, A = 1, omega = 1, phi = 1):
        return A * np.sin(omega * t + phi)

    def Cosinusoidal(x, t, A = 1, omega = 1, phi = 1):
        return A * np.cos(omega * t + phi)

    def PlotBV(B, V, plotName):
        plt.figure()
        
        fig, ax = plt.subplots()
        ax.invert_yaxis()

        plt.plot(unp.nominal_values(V), unp.nominal_values(B - V), "kx")

        plt.xlabel("V")
        plt.ylabel("B - V")
        
        plt.savefig(plotName)  