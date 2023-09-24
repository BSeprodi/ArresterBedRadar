# Author: Seprodi Barnabas
# Version: 2.1
# Date: 2023-07-02

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptch
import time

""" Constants: """
CAMERA_MATRIX = np.array([
    [271.85233228, 0, 311.91024189],
    [0, 271.78293015, 175.03290332],
    [0, 0, 1]
])
DISTORTION_CF = np.array([
    [-0.10869424, 0.00628939, 0.00207044, 0.00326306, 0.00237423]
])

""" Functions: """
# Require valid input
def require(prompt,f):
    while True:
        try:
            return f(input(prompt))
        except ValueError:
            print("\033[F\033[K[!] ", end="")
        except KeyboardInterrupt:
            print()
            break

def undistortFrame(img,cameraMatrix,distortionCoeff):
    h,w,*_ = img.shape
    newCameraMatrix, ROI = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortionCoeff, (w,h), 1, (w,h))
    return cv2.undistort(img,cameraMatrix,distortionCoeff,None,newCameraMatrix)

def filterBGR(img,lower,upper=[255,255,255]):
    return cv2.inRange(img,np.array(lower),np.array(upper))

def averageFiltered(img):
    h,w,*_ = img.shape
    weight = np.arange(0,h)
    return np.array([np.sum(weight * column / 255) / (np.count_nonzero(column) + 1e-10) for column in img.T])

def drawBrightestLine(img,pos):
    for i,p in enumerate(pos):
        img[int(p),i] = [255,0,0]

def drawCross(img):
    h,w,*_ = img.shape
    cv2.line(img,(0,h//2),(w,h//2),(0,0,255),1)
    cv2.line(img,(w//2,0),(w//2,h),(0,0,255),1)

def drawSquare(img):
    h,w,*_ = img.shape
    cv2.rectangle(img,(w//4,h//4),(3*w//4,3*h//4),(0,0,255),2)

# Convert pixel coordinates to actual coordinates
def pos2depth(position,center,D,H):
    Z = 0.0026511 # scaling coefficient
    radians = (center - position) * np.pi/9 / np.tan(np.pi/9) * Z
    return H - D * (H/D - np.tan(radians)) / (1+ H/D *np.tan(radians))

# Delete null values 
def clean(X,Z,D,H,epsilon=1e-2):
    null = pos2depth(0,180,D,H)
    notNull = [i for i,v in enumerate(Z) if abs(null - v) > epsilon]
    return X[notNull], Z[notNull]

# Calculate width height and area
def stats(X,Z):
    h = min(Z)
    m = i = j = np.where(Z == h)[0][0]
    while i > 0 and Z[i] < 0: i -= 1
    while j < len(Z)-1 and Z[j] < 0: j += 1
    w = X[j] - X[i]
    A = sum([(X[k+1] - X[k]) * (Z[k] + Z[k+1]) / 2 for k in range(i,j)])
    return w,h,A

# Import points from csv and sort by y then x
def importCsv(path):
    data = []
    with open(path,"r") as f:
        for line in f:
            x,y,z = map(float, line.replace(",",".").split(";"))
            data.append([x,y,z])
        f.close()
    data = np.array(data).T
    ind = np.lexsort((data[0],data[1]))
    return data[0][ind], data[1][ind], data[2][ind]

""" Main script: """
def main():
    # Initial values
    T_start = 0
    T_end = 0
    filename = time.strftime("%b%d-%H%M")
    D = 0.2
    H = 0.1
    B = 255

    print("   _                  _           ___         _ ___         _          ")
    print("  /_\  _ _ _ _ ___ __| |_ ___ _ _| _ ) ___ __| | _ \__ _ __| |__ _ _ _ ")
    print(" / _ \| '_| '_/ -_|_-<  _/ -_) '_| _ \/ -_) _` |   / _` / _` / _` | '_|")
    print("/_/ \_\_| |_| \___/__/\__\___|_| |___/\___\__,_|_|_\__,_\__,_\__,_|_|  ")
    print("------------------- Seprodi Barnabas | Version: 2.1 -------------------")
    print(f"Defaults: {filename = }, {D = }m, {H = }m, {B = }")

    CAP = cv2.VideoCapture(1) # For USB camera use 1 as value
    CAP.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Set width
    CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # Set height

    while True:
        # Read frame from camera
        _, frame = CAP.read()
        frame = frame[::-1] # Flip view

        # Compensate for 'fisheye' effect
        frame = undistortFrame(frame, CAMERA_MATRIX, DISTORTION_CF)

        # Retrieve brightest areas
        mask = filterBGR(frame, [B,B,B], [255,255,255])

        # Average brightest areas along vertical direction
        pos = averageFiltered(mask) 

        # Calculate displacement in X and Z direction
        X = pos2depth(np.arange(640,0,-1),320,D,0)
        Z = pos2depth(pos,180,D,H)

        # Remove null values
        X,Z = clean(X,Z,D,H)

        # Print parameters to screen
        drawBrightestLine(frame, pos)
        drawCross(frame)
        drawSquare(frame)
        cv2.putText(frame,f"{D = }m",(485,100),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
        cv2.putText(frame,f"{H = }m",(485,120),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
        cv2.putText(frame,f"{B = }",(485,140),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255))
        
        KEY = cv2.waitKey(1)

        # Record
        if T_start < time.time() and T_end >= time.time():
            T = float(time.time() - T_start)
            cv2.putText(frame, f"Rec.: {round(T,2)}", (485, 160), cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0))
            with open(f"measurements/{filename}.csv","a+") as f:
                for i in range(len(X)):
                    f.write(f"{X[i]}; {T}; {Z[i]}\n".replace(".",","))
                f.close()
            # Create stats (order: y coordinate, width, maximal depth)
            with open(f"measurements/{filename}-stats.csv","a+") as f:
                w,h,A = stats(X,Z)
                f.write(f"{T}; {w}; {h}; {A}\n".replace(".",","))
                f.close()

        # Quit (ESC)
        elif KEY == 27:
            cv2.destroyAllWindows()
            print("Quit ------------------------------------------------------------------")
            break

        # Change parameters (c)
        elif KEY == ord("c"):
            cv2.destroyAllWindows()
            print("Change parameters -----------------------------------------------------")
            D = require(f"{D = }m -> ",float)
            H = require(f"{H = }m -> ",float)

        # Change filename (f)
        elif KEY == ord("f"):
            cv2.destroyAllWindows()
            print("Change filename -------------------------------------------------------")
            filename = require(f"{filename = } -> ",str)
            
        # Change brightness (b)
        elif KEY == ord("b"):
            cv2.destroyAllWindows()
            print("Change brightness -----------------------------------------------------")
            B = require(f"{B = } -> ",int)

        # Save (s)
        elif KEY == ord("s"):
            cv2.destroyAllWindows()
            Y = require("Y (m) = ",float)
            # Save points (order: x,y,z coordinate)
            with open(f"measurements/{filename}.csv","a+") as f:
                for i in range(len(X)):
                    f.write(f"{X[i]}; {Y}; {Z[i]}\n".replace(".",","))
                f.close()
            # Create stats (order: y coordinate, width, maximal depth)
            with open(f"measurements/{filename}-stats.csv","a+") as f:
                w,h,A = stats(X,Z)
                f.write(f"{Y}; {w}; {h}; {A}\n".replace(".",","))
                f.close()
        
        # Preview (p)
        elif KEY == ord("p"):
            xcor = pos2depth(160,320,D,H)
            ycor = pos2depth(90,180,D,0)
            fig, ax = plt.subplots()
            ax.grid()
            ax.scatter(X,Z)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
            ax.add_patch(ptch.Rectangle((-xcor, -ycor),2*xcor,2*ycor,linewidth=1, edgecolor="r",facecolor="none"))
            plt.show()
        
        # Help (h)
        elif KEY == ord("h"):
            print("Help ------------------------------------------------------------------")
            print("Options:")
            print("b\tChange brigthness level for masking")
            print("c\tChange parameters")
            print("f\tChange filename")
            print("h\tHelp")
            print("p\tShow preview")
            print("r\tRecord video")
            print("s\tSave frame")
            print("v\tVisualize saved points")
            print("ESC\tQuit")

            # print(" Options: ESC (quit), c (change parameters), f (change filename), b (change B), s (save), p (preview), h (help)")
            # print(" '[!]' means nvalid value, try again (e.g. when float value is expected, but recieved string)")
            # print(f" When saving '{filename}' contains x,y,z coordinates, '{filename}-stats' contains y coordinate, width, depth")  
            # print(" To convert old csv files use 'converter.py'.")

        # Record video (r)
        elif KEY == ord("r"):
            cv2.destroyAllWindows()
            print("Duration --------------------------------------------------------------")
            deltaT = require("T (s) = ", float)
            T_start = time.time() + 3
            T_end = T_start + deltaT
            print("[!] Starting in 3 seconds...")

        # Visualize (v)
        elif KEY == ord("v"):
            try:
                x,y,z = importCsv(f"measurements/{filename}.csv")
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(x,y,z, marker=".")
                ax.set_xlabel("x [m]")
                ax.set_ylabel("t [s]")
                ax.set_zlabel("z [m]")
                plt.show()
            except FileNotFoundError:
                print("[!] File does not exist")

        cv2.imshow("Camera", frame)

if __name__ == "__main__":
    main()
