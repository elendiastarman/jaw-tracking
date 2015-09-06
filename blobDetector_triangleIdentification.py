import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, cos, sin, pi
from time import clock
from copy import deepcopy

def getVideoFrame(VIDEO_PATH, xdim=800, ydim=600, frame=60):
    FFMPEG_BIN = r"C:\Python33\Lib\ffmpeg-20140713-git-42c1cc3-win64-static\bin\ffmpeg.exe"

    ### get video info
    import subprocess as sp
    command = [FFMPEG_BIN, '-i', VIDEO_PATH, '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
    pipe.stdout.readline()
    pipe.terminate()
    infos = str(pipe.stderr.read())[2:-1]

    ### get the frames per second
    end = infos.index("fps")-1
    beg = end-1
    while infos[beg-1] != " ":
        beg -= 1
    fps = float(infos[beg:end])

    ### pick the desired frame
    Hz = fps

    ### calculate timecode from frame and fps
    s = (frame // Hz) % 60
    m = (frame // Hz) // 60 % 60
    h = (frame // Hz) // 3600
    f = (frame %  Hz) * Hz
    timecode = "%02d:%02d:%02d.%03d" % (h,m,s,f)

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

    ### dimensions of image
##    xdim = 800
##    ydim = 600

    ### get frame image
    raw_image = pipe.stdout.read(xdim*ydim*3)
    image = np.fromstring(raw_image, dtype='uint8')
    image = image.reshape((ydim,xdim,3))
    pipe.stdout.flush()
    pipe.terminate()

    return image, xdim,ydim, frame

def colorDistance(target,actual): return sqrt(sum([(a-b)**2 for a,b in zip(target,actual)]))
def colorDistanceMagSquared(target,actual): return sum([(a-b)**2 for a,b in zip(target,actual)])
def colorDistanceSum(target,actual): return sum([abs(a-b) for a,b in zip(target,actual)])
def colorIsExact(target,actual): return target == actual

def colDis(targetColor): return lambda col: colorDistance(targetColor,col)
def colDisMagSq(targetColor): return lambda col: colorDistanceMagSquared(targetColor,col)
def colDisSum(targetColor): return lambda col: colorDistanceSum(targetColor,col)
def colIsExact(targetColor): return lambda col: colorIsExact(targetColor,col)

def gray(z): return [np.uint8(z)]*3
def col(r,g,b): return list(map(np.uint8,[r,g,b]))
def mean(X): return sum(X)/len(X)

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

        yield sx+dx*spacing, sy+dy*spacing

def identifyTriangles(points, debug=0):
    #precondition: all points are accounted for in exactly two triangles
    if len(points) != 6:
        raise ValueError("Too many or too few points. Need exactly 6.")

    tris = []

    n = len(points)
    def D(x1,y1, x2,y2): return sqrt((x2-x1)**2+(y2-y1)**2)

    for k1 in range(n-2):
        p1 = points[k1]
        for k2 in range(k1+1,n-1):
            if k1 != k2:
                p2 = points[k2]
                for k3 in range(k2+1,n):
                    if k3 != k1 and k3 != k2:
                        p3 = points[k3]

                        d1 = D(p1[0],p1[1], p2[0],p2[1])
                        d2 = D(p1[0],p1[1], p3[0],p3[1])
                        d3 = D(p2[0],p2[1], p3[0],p3[1])

                        mean = round((d1+d2+d3)/3, 3) #arithmetic mean
                        var = round(sum([(d-mean)**2 for d in [d1,d2,d3]])/3, 3) #variance

                        tris.append([var, mean, [p1,p2,p3]])

    tris.sort()
    if debug:
        for t in tris: print(t)

    def overlap(L1, L2):
        L1s = sorted(L1)
        L2s = sorted(L2)

        i,j = 0,0
        while i < len(L1s) and j < len(L2s):
            if L1s[i] == L2s[j]: return 1
            if   L1s[i] < L2s[j]: i += 1
            elif L1s[i] > L2s[j]: j += 1

        return 0

    m = 0
    mi,mj = 0, 0

    def mean(X): return sum(X)/len(X)
    def dot(X,Y): return sum(a*b for a,b in zip(X,Y))
    def diff(X,Y): return [y-x for x,y in zip(X,Y)] #\vec{Y} - \vec{X}

    for i in range(5):
        for j in range(1,6):
            if not overlap(tris[i][2], tris[j][2]):
                triCenter1 = list(map(mean, zip(*tris[i][2])))
                triCenter2 = list(map(mean, zip(*tris[j][2])))
                centerDiff = diff(triCenter1, triCenter2)

                tri1m = max([dot(centerDiff, diff(triCenter1, p)) for p in tris[i][2]])
                tri2m = max([dot(centerDiff, diff(triCenter2, p)) for p in tris[j][2]])

                if tri1m + tri2m > m:
                    m = tri1m + tri2m
                    mi,mj = i,j

    return [tris[mi], tris[mj]]



### SEARCH ###

def search_squareSpiral(blobs, xdim, ydim, spacing=1):
    startX = xdim//2
    startY = ydim//2

    spiral = squareSpiral(startX,startY, spacing, min(xdim,ydim)//(2*spacing)-1)

    while 1:

        try:
            centerX, centerY = next(spiral)
        except StopIteration:
            return

        goodEnough = (cD(image[centerY][centerX]) <= maxColDis)

        if goodEnough:
            dupe = 0
            for blob in blobs:
                if (blob[0]-centerX)**2 + (blob[1]-centerY)**2 <= blob[2]**2*1.5:
                    dupe = 1
                    break

            if not dupe:
                yield centerX, centerY


### SURVEY ###

def survey_floodFill(sX,sY):
    cX,cY,cR = (None,None,None)

    queue = [[sX,sY]]
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

    return (cX,cY,cR)

def survey_circumcircle(sX,sY, numIterations=3, showVerts=0): #showVerts is also "minRadius"
    cX,cY,cR = (None,None,None)
    
    for k in range(numIterations): #refines circumcircle estimate by using prior guess
        vertices = []

        for dx,dy in [[1,0],[0,1],[-1,0],[0,-1]]:
            pX = sX+dx
            pY = sY+dy
            while cD(image[pY][pX]) <= maxColDis:
                pX += dx
                pY += dy
            vertices.append( [pX,pY] )

        v = vertices

        #get four possible circle centers/radii
        c1 = circumcircle(v[0],v[1],v[2])
        c2 = circumcircle(v[0],v[1],v[3])
        c3 = circumcircle(v[0],v[2],v[3])
        c4 = circumcircle(v[1],v[2],v[3])

        #average them
        avgX = (c1[0]+c2[0]+c3[0]+c4[0])/4
        avgY = (c1[1]+c2[1]+c3[1]+c4[1])/4
        avgR = (c1[2]+c2[2]+c3[2]+c4[2])/4

        c5 = [avgX,avgY,avgR]
        cX,cY,cR = c5

        sX = cX
        sY = cY


    if showVerts and cR >= showVerts:
        image[cY][cX] = col(0,0,255)
        for vert in vertices:
            image[vert[1]][vert[0]] = col(255,0,0)

    return (round(cX),round(cY),round(cR))



### The actual start of the program ###

VIDEO_PATH = r".\Demo vids 1\demovid1_left0001-0075.mp4"

targetColor = [0,255,0]
cD = colDisSum(targetColor)
minRadius = 6 #pixels
maxColDis = 40

origImage, xdim,ydim, frame = getVideoFrame(VIDEO_PATH)

### Search algorithm entries go like this:
### ['name', function_name, <optional parameter 1>, <opt. param. 2>, <etc>]
### Functions are expected to take -blobs, xdim, ydim- as their first three args
### They must -yield- some -sx, sy- to survey

search_algs = [#['pixel by pixel', search_pixels],
               #['pixels spaced apart', search_spacedPixels, minRadius],
               ['square spiral', search_squareSpiral, minRadius],
               #['rectangular quadsection', search_rectQuadsection, minRadius],
               ]

### Survey algorithm entries go like this:
### ['name', function_name, <optional parameter 1>, <opt. param. 2>, <etc>]
### Functions are expected to take -sx, sy- as their first two args
### They must -return- a tuple: -(cX,cY,cR)-; that is, blob x,y, and radius

survey_algs = [#['flood fill', survey_floodFill],
               ['circumcircle', survey_circumcircle],
               ]

##plt.imshow(origImage)

n = 0
for search_alg in search_algs:
    for survey_alg in survey_algs:

        n += 1
        image = deepcopy(origImage) #ensures that flood fill doesn't mess up other algorithms

        fig = plt.figure(n)
        sub = fig.add_subplot(111)
        sub.imshow(image)

        blobs = [] #for storing blobs
        search = search_alg[1](blobs, xdim,ydim, *search_alg[2:])

        while 1:
            try:
                sx,sy = next(search)
            except StopIteration:
                break

            blob = survey_alg[1](sx,sy, *survey_alg[2:])
            if blob[2] >= minRadius: blobs.append(blob)

        if len(blobs) != 6:
            print("Uh oh! Not enough or too many blobs were detected!")
            break

        tris = identifyTriangles(blobs)

        for t in tris:
            for k in range(3):
                sub.plot([t[2][k-1][0],t[2][k][0]],
                         [t[2][k-1][1],t[2][k][1]],
                         'b-')

        sub.set_xlim([0,xdim])
        sub.set_ylim([ydim,0])

plt.show()
