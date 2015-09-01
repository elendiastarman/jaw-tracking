FFMPEG_BIN = r"C:\Python33\Lib\ffmpeg-20140713-git-42c1cc3-win64-static\bin\ffmpeg.exe"
VIDEO_PATH = r".\Demo vids 1\demovid1_left0001-0075.mp4"
#r"C:\Users\El'endia Starman\Desktop\My works\Jaw Tracking\Demo vids 1\demovid1_left0001-0075.mp4"

### get video info
import subprocess as sp
command = [FFMPEG_BIN, '-i', VIDEO_PATH, '-']
pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
pipe.stdout.readline()
pipe.terminate()
infos = str(pipe.stderr.read())[2:-1]
print(infos)

### get the frames per second
end = infos.index("fps")-1
beg = end-1
while infos[beg-1] != " ":
    beg -= 1
fps = float(infos[beg:end])

### pick the desired frame
frame = 60
Hz = fps#round(fps)

### calculate timecode from frame and fps
s = (frame // Hz) % 60
m = (frame // Hz) // 60 % 60
h = (frame // Hz) // 3600
f = (frame %  Hz) * Hz
timecode = "%02d:%02d:%02d.%03d" % (h,m,s,f)
print(timecode)

### set up pipe to get single video frame
command = [ FFMPEG_BIN,
            '-ss', timecode,
            '-i', VIDEO_PATH,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo', '-']
pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

### get and format single video frame
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, pi
from time import clock

### dimensions of image
xdim = 800
ydim = 600

### get frame image
raw_image = pipe.stdout.read(xdim*ydim*3)
image = np.fromstring(raw_image, dtype='uint8')
image = image.reshape((ydim,xdim,3))
pipe.stdout.flush()
pipe.terminate()

### display image
fig2 = plt.figure(2)
#plt.imshow(image)

targetColor = [0,255,0]
def colorDistance(target,actual): return sqrt(sum([(a-b)**2 for a,b in zip(target,actual)]))
def colDis(targetColor):
    return lambda col: colorDistance(targetColor,col)
def gray(z): return [np.uint8(z)]*3
def col(r,g,b): return list(map(np.uint8,[r,g,b]))

def circumcircle(p1,p2,p3): #each point is expected to be [x,y]
    #taken from https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_from_cross-_and_dot-products
    diff1 = [p1[0]-p2[0], p1[1]-p2[1]]
    diff2 = [p1[0]-p3[0], p1[1]-p3[1]]
    diff3 = [p2[0]-p3[0], p2[1]-p3[1]]
    
    magSquared1 = diff1[0]**2 + diff1[1]**2
    magSquared2 = diff2[0]**2 + diff2[1]**2
    magSquared3 = diff3[0]**2 + diff3[1]**2

    mag1 = sqrt(magSquared1)
    mag2 = sqrt(magSquared2)
    mag3 = sqrt(magSquared3)

    dot1 = diff1[0]*diff2[0] + diff1[1]*diff2[1]
    dot2 = -diff1[0]*diff3[0] + -diff1[1]*diff3[1]
    dot3 = -diff2[0]*-diff3[0] + -diff2[1]*-diff3[1]

    #working in 2D simplifies the cross product and its magnitude a lot
    crossMag = diff1[0]*diff3[1] - diff1[1]*diff3[0]
    crossMagSquared = crossMag**2

    r = mag1*mag2*mag3/(2*crossMag)

    alpha = magSquared3*dot1/(2*crossMagSquared)
    beta  = magSquared2*dot2/(2*crossMagSquared)
    gamma = magSquared1*dot3/(2*crossMagSquared)

    x = alpha*p1[0] + beta*p2[0] + gamma*p3[0]
    y = alpha*p1[1] + beta*p2[1] + gamma*p3[1]

    return [x,y,r]

def squareSpiral(startX,startY, spacing, limit):
    sx = startX
    sy = startY
    yield sx,sy

    level = 0
    dx = -1
    dy = 0

    while level < limit:
        if dx == -1 and dy == level:
            level += 1
            dx = 0
            dy = level
        else:
            if dy == level and dx < level:
                dx += 1
            elif dx == level and dy > -level:
                dy -= 1
            elif dy == -level and dx > -level:
                dx -= 1
            elif dx == -level and dy < level:
                dy += 1

##        print(dx,dy,level)

        yield sx+dx*spacing, sy+dy*spacing


cD = colDis(targetColor)
minRadius = 6 #pixels
maxColDis = 40

print("Ready to start!")
startTime = clock()

theta = 0
rad = minRadius
startX = xdim//2
startY = ydim//2
radLimit = min(xdim,ydim)

numIterations = 3

blobs = [] #blobs will be stored as [x,y,r]
spiral = squareSpiral(startX,startY, minRadius, min(xdim,ydim)//(2*minRadius)-1)

while 1:
##    quad = quads.pop(0)
##    centerX = (quad[0][0]+quad[1][0])//2
##    centerY = (quad[0][1]+quad[1][1])//2
    #image[centerY][centerX] = col(55,0,0)

    try:
        centerX, centerY = next(spiral)
    except StopIteration:
        break
    
##    image[centerY][centerX] = col(255,0,0)
##
##    continue

    if cD(image[centerY][centerX]) <= maxColDis:
        dupe = 0
        for blob in blobs:
            if (blob[0]-centerX)**2 + (blob[1]-centerY)**2 <= blob[2]**2*1.5:
                dupe = 1
                break

        if not dupe:
            image[centerY][centerX] = col(255,0,0)
            
            queue = [[centerX,centerY]]
            q = 0
            while q < len(queue):
                qx,qy = queue[q]
                
                for dx,dy in [[1,0],[0,1],[-1,0],[0,-1]]:
                    x2, y2 = qx+dx, qy+dy

                    if cD(image[y2][x2]) <= maxColDis:
                        image[y2][x2] = col(255,0,0)
                        queue.append([x2,y2])

                q += 1

            sumX,sumY = list(map(sum, zip(*queue)))
            cX, cY = round(sumX/q), round(sumY/q)
            cR = round(sqrt( q/pi ))

            image[cY][cX] = col(0,0,255)

            blobs.append([cX,cY,cR])

endTime = clock()
print("All done in %.3f seconds!" % (endTime-startTime))

plt.imshow(image)
plt.show()
