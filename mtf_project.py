import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector
import numpy as np
import cv2
import argparse
from PIL import Image
from scipy import interpolate
from scipy.fft import fft
from dataclasses import dataclass

@dataclass
class cSet:
    x: np.ndarray
    y: np.ndarray

@dataclass
class cESF: 
    rawESF: cSet
    interpESF: cSet
    threshold: float
    width: float
    angle: float
    edgePoly: np.ndarray

@dataclass
class cMTF: 
    x: np.ndarray
    y: np.ndarray
    mtfAtNyquist: float
    width: float

class Helper:
    @staticmethod
    def LoadImage(filename):  
        img = Image.open(filename)
        if img.mode in {'I;16','I;16L','I;16B','I;16N'}:
            gsimg = img
        else:
            gsimg = img.convert('L')
        return gsimg

    @staticmethod
    def LoadImageAsArray(filename): 
        img = Helper.LoadImage(filename)
        if img.mode in {'I;16','I;16L','I;16B','I;16N'}:
            arr = np.asarray(img, dtype=np.double)/65535
        else:
            arr = np.asarray(img, dtype=np.double)/255
        return arr

    @staticmethod
    def CorrectImageOrientation(imgArr):  
        tl = np.average(imgArr[0:2,0:2])
        tr = np.average(imgArr[0:2,-3:-1])
        bl = np.average(imgArr[-3:-1,0:2])
        br = np.average(imgArr[-3:-1,-3:-1])
        edges = [tl, tr, bl, br]
        edgeIndexes = np.argsort(edges)
        if (edgeIndexes[0] + edgeIndexes[1]) == 1:
            pass
        elif (edgeIndexes[0] + edgeIndexes[1]) == 5:
            imgArr = np.flip(imgArr, axis=0)
        elif (edgeIndexes[0] + edgeIndexes[1]) == 2:
            imgArr = np.transpose(imgArr)
        elif (edgeIndexes[0] + edgeIndexes[1]) == 4:
            imgArr = np.flip(np.transpose(imgArr), axis=0)
        return imgArr

class MTF:
    @staticmethod
    def SafeCrop(values, distances, head, tail):
        isIncrementing = True
        if distances[0] > distances[-1]:
            isIncrementing = False
            distances = -distances
            dummy = -tail
            tail = -head
            head = dummy
        hindex = (np.where(distances < head)[0])
        tindex = (np.where(distances > tail)[0])
        if hindex.size < 2:
            h = 0
        else:
            h = np.amax(hindex)
        if tindex.size == 0:
            t = distances.size
        else:
            t = np.amin(tindex)
        if isIncrementing == False:
            distances = -distances
        return cSet(distances[h:t], values[h:t])

    @staticmethod
    def GetEdgeSpreadFunction(imgArr, edgePoly):
        Y = imgArr.shape[0]
        X = imgArr.shape[1]
        values = np.reshape(imgArr, X*Y)
        distance = np.zeros((Y,X))
        column = np.arange(0,X)+0.5
        for y in range(Y):
            distance[y,:] = (edgePoly[0]*column - (y+0.5) + edgePoly[1]) / np.sqrt(edgePoly[0]*edgePoly[0] + 1)
        distances = np.reshape(distance, X*Y)
        indexes = np.argsort(distances)
        sign = 1
        if np.average(values[indexes[:10]]) > np.average(values[indexes[-10:]]):
            sign = -1
        values = values[indexes]
        distances = sign*distances[indexes]     
        if (distances[0] > distances[-1]):
            distances = np.flip(distances)
            values = np.flip(values)
        return cSet(distances, values)

    @staticmethod
    def GetEdgeSpreadFunctionCrop(imgArr):
        imgArr = Helper.CorrectImageOrientation(imgArr)
        edgeImg = cv2.Canny(np.uint8(imgArr*255), 40, 90, L2gradient=True)
        line = np.argwhere(edgeImg == 255)
        if line.size == 0:
            raise ValueError("No edge detected in the selected ROI. Please select a region with a clear edge.")
        edgePoly = np.polyfit(line[:,1],line[:,0],1)
        angle = np.degrees(np.arctan(-edgePoly[0]))
        finalEdgePoly = edgePoly.copy()
        if angle > 0:
            imgArr = np.flip(imgArr, axis=1)
            finalEdgePoly[1] = np.polyval(edgePoly,np.size(imgArr,1)-1)
            finalEdgePoly[0] = -edgePoly[0]
        esf = MTF.GetEdgeSpreadFunction(imgArr, finalEdgePoly)
        esfValues = esf.y
        esfDistances = esf.x
        maximum = np.amax(esfValues)
        minimum = np.amin(esfValues)
        threshold = (maximum - minimum) * 0.1
        head = np.amax(esfDistances[(np.where(esfValues < minimum + threshold))[0]])
        tail = np.amin(esfDistances[(np.where(esfValues > maximum - threshold))[0]])
        width = abs(head-tail)
        esfRaw = MTF.SafeCrop(esfValues, esfDistances, head - 1.2*width, tail + 1.2*width)
        qs = np.linspace(0,1,20)[1:-1]
        knots = np.quantile(esfRaw.x, qs)
        tck = interpolate.splrep(esfRaw.x, esfRaw.y, t=knots, k=3)
        ysmooth = interpolate.splev(esfRaw.x, tck)
        InterpDistances = np.linspace(esfRaw.x[0], esfRaw.x[-1], 500)
        InterpValues = np.interp(InterpDistances, esfRaw.x, ysmooth)
        esfInterp = cSet(InterpDistances, InterpValues)
        return cESF(esfRaw, esfInterp, threshold, width, angle, edgePoly)

    @staticmethod
    def GetLineSpreadFunction(esf, normalize=True):
        lsfDividend = np.diff(esf.y)
        lsfDivisor = np.diff(esf.x)
        lsfValues = np.divide(lsfDividend, lsfDivisor)
        lsfDistances = esf.x[0:-1]
        if normalize:
            lsfValues = lsfValues / (max(lsfValues))
        return cSet(lsfDistances, lsfValues)

    @staticmethod
    def GetMTF(lsf):
        N = np.size(lsf.x)
        px = N / (lsf.x[-1] - lsf.x[0])
        values = 1/np.sum(lsf.y) * abs(fft(lsf.y))
        distances = np.arange(0, N) / N * px
        interpDistances = np.linspace(0, 1, 200)
        interp = interpolate.interp1d(distances, values, kind='cubic')
        interpValues = interp(interpDistances)
        interpValues /= np.max(interpValues)
        mtf50_freq = np.interp(0.5, interpValues[::-1], interpDistances[::-1])
        return cMTF(interpDistances, interpValues, mtf50_freq, -1.0)


    @staticmethod
    def CalculateMtf(filename, imgArr, roi=None):
        if roi is not None:
            roi = roi.astype(int)
            imgArr = imgArr[roi[0]:roi[1], roi[2]:roi[3]]
        try:
            esf = MTF.GetEdgeSpreadFunctionCrop(imgArr)
        except ValueError as e:
            print(e)
            return None
        lsf = MTF.GetLineSpreadFunction(esf.interpESF, True)
        mtf = MTF.GetMTF(lsf)
        
        # Visualization in the style of mtf_incorrect.py
        fig = plt.figure(figsize=(10, 8))
        roi_str = f"ROI: [{roi[0]}:{roi[1]}, {roi[2]}:{roi[3]}]" if roi is not None else "Full Image"
        fig.suptitle(f"{filename} MTF Analysis\n{roi_str}", fontsize=20)
        
        # Subplot 1: ROI with fitted edge line
        plt.subplot(2, 2, 1)
        plt.imshow(imgArr, cmap='gray')
        plt.title("Selected ROI")

        # Subplot 2: ESF
        plt.subplot(2, 2, 2)
        plt.title("ESF Curve")
        plt.xlabel("Pixel")
        plt.ylabel("Intensity")
        # plt.plot(esf.rawESF.x, esf.rawESF.y, 'y-', label='Raw ESF')
        plt.plot(esf.interpESF.x, esf.interpESF.y, 'b-', label='Smooth ESF')
        plt.legend(handles=[mpatches.Patch(color='blue', label='Smooth ESF')])
 
        # Subplot 3: LSF
        plt.subplot(2, 2, 3)
        plt.title("LSF Curve")
        plt.xlabel("Pixel")
        plt.ylabel("Intensity")
        plt.plot(lsf.x, lsf.y, 'y-', label='Raw LSF')
        plt.legend(handles=[mpatches.Patch(color='yellow', label='Raw LSF')])
        
        # Subplot 4: MTF
        plt.subplot(2, 2, 4)
        plt.title(f"MTF Curve (MTF50: {mtf.mtfAtNyquist:.3f} cycles/pixel)")
        plt.xlabel("Cycles/Pixel")
        plt.ylabel("Modulation Factor")
        plt.plot(mtf.x, mtf.y, 'b-', label='MTF')
        plt.legend(handles=[mpatches.Patch(color='blue', label='MTF')])
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        
        print(f"MTF50: {mtf.mtfAtNyquist:.3f} cycles/pixel")
        print(f"Transition Width: {esf.width:.2f} pixels")
        return mtf

class EventHandler:
    def __init__(self, filename, imgArr):
        self.filename = filename
        self.imgArr = imgArr
        self.roi = None

    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        self.roi = np.array([int(min(y1, y2)), int(max(y1, y2)), int(min(x1, x2)), int(max(x1, x2))])

    def event_exit_manager(self, event):
        if event.key == 'enter' and self.roi is not None:
            plt.close()
            MTF.CalculateMtf(self.filename, self.imgArr, self.roi)

class ROI_selection:
    def __init__(self, filename):
        self.filename = filename
        self.imgArr = Helper.LoadImageAsArray(filename)
        fig, ax = plt.subplots()

        ax.imshow(self.imgArr, cmap='gray')
        ax.axis("off") 
        ax.set_frame_on(False)

        plt.title("Select ROI and press Enter")

        eh = EventHandler(self.filename, self.imgArr)
        rectangle_selector = RectangleSelector(
            ax,
            eh.line_select_callback,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            interactive=True
        )
        plt.connect('key_press_event', eh.event_exit_manager)
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', help='String Filepath')
    args = parser.parse_args()
    filename = args.filepath
    ROI_selection(filename)