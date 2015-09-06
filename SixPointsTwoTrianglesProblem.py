from random import randint, shuffle
from math import sin, cos, sqrt, pi

def generatePoints(numPolys = 2, numVerts = 3, scramble=1, debug=0, **kwargs):
    points = []

    xMin = -100 if 'xMin' not in kwargs else kwargs['xMin']
    xMax =  100 if 'xMax' not in kwargs else kwargs['xMax']
    yMin = -100 if 'yMin' not in kwargs else kwargs['yMin']
    yMax =  100 if 'yMax' not in kwargs else kwargs['yMax']

    angMin = -5 if 'angMin' not in kwargs else kwargs['angMin']
    angMax =  5 if 'angMax' not in kwargs else kwargs['angMax']

    rad = 40 if 'rad' not in kwargs else kwargs['rad']
    radVar = 5 if 'radVar' not in kwargs else kwargs['radVar']
    posVar = 5 if 'posVar' not in kwargs else kwargs['posVar']
##    angVar =  5 if 'angVar' not in kwargs else kwargs['angVar']

    for p in range(numPolys):

        cx = randint(xMin,xMax)
        cy = randint(yMin,yMax)
        ang = randint(angMin,angMax)
        r = rad + randint(-radVar,radVar)

        angstep = 2*pi/numVerts
        
        for v in range(numVerts):
            while 1: #bit of a hack to ensure no points are duplicated
                vx = cx + r*cos(ang + v*angstep) + randint(-posVar,posVar)
                vy = cy + r*sin(ang + v*angstep) + randint(-posVar,posVar)

                vx = round(vx,1)
                vy = round(vy,1)
                
                if [vx,vy] not in points:
                    if debug: print("Poly %d, vertex %d: (%s,%s)" % (p,v,vx,vy) )
                    points.append([vx,vy])
                    break

    if scramble: shuffle(points)
    return points

def D(x1,y1, x2,y2):
    return sqrt((x2-x1)**2+(y2-y1)**2)

##def mean(X): return sum(X)/len(X)
##def stddev(X):
##    xbar = mean(X)
##    return sum([(x-xbar)**2 for x in X])/len(X)

def getAffine(initialVec1,initialVec2, finalVec1,finalVec2, initialVec3=None,finalVec3=None):
    if initialVec3 and finalVec3: #need translation vector
        initialVec1[0] -= initialVec3[0] #this is possible because I don't need the true initialVec1 anymore
        initialVec1[1] -= initialVec3[1]
        initialVec2[0] -= initialVec3[0]
        initialVec2[1] -= initialVec3[1]
        finalVec1[0] -= finalVec3[0] #this is possible because I don't need the true initialVec1 anymore
        finalVec1[1] -= finalVec3[1]
        finalVec2[0] -= finalVec3[0]
        finalVec2[1] -= finalVec3[1]

    x1 = initialVec1[0]
    y1 = initialVec1[1]
    x2 = initialVec2[0]
    y2 = initialVec2[1]

    u1 = finalVec1[0]
    v1 = finalVec1[1]
    u2 = finalVec2[0]
    v2 = finalVec2[1]

    sharedDenom = y2*x1-y1*x2
    
    a = (y2*u1-y1*u2)/sharedDenom
    b = (x2*u1-x1*u2)/-sharedDenom
    c = (y2*v1-y1*v2)/sharedDenom
    d = (x2*v1-x1*v2)/-sharedDenom

    if initialVec3 and finalVec3:
        e = finalVec3[0] - a*initialVec3[0] - b*initialVec3[1]
        f = finalVec3[1] - c*initialVec3[0] - d*initialVec3[1]
        
        return [[a,b,e],[c,d,f]]
    else:
        return [[a,b],[c,d]]

def findPolys(points, numPolys = 2, numVerts = 3):
    #precondition: all points are accounted for in exactly -numPolys- polygons
    if len(points) != numPolys*numVerts:
        raise ValueError("Too many or too few points. %d != %d*%d" % (len(points), numPolys, numVerts))
    elif len(points) <= 2:
        raise ValueError("Too few points. Need at least 3.")

    polys = []

    triplets = []
    n = len(points)

    for k1 in range(n):
        p1 = points[k1]
        for k2 in range(n-1):
            if k1 != k2:
                p2 = points[k2]
                for k3 in range(k2+1,n):
                    if k3 != k1 and k3 != k2:
                        p3 = points[k3]

                        d1 = D(p1[0],p1[1], p2[0],p2[1])
                        d2 = D(p1[0],p1[1], p3[0],p3[1])

                        mean = round((d1+d2)/2, 2) #arithmetic mean
                        sigma = round(abs(d1-d2)/2, 2) #standard deviation of two numbers

                        xi = round(mean*sigma, 2)

                        triplets.append( [xi, mean, sigma, [p1,p2,p3] ] )

    triplets.sort()

    for t in triplets: print(t)

    #construct polys from triplets here

    return polys

def getTwoTriangles(points, debug=0):
    #precondition: all points are accounted for in exactly two triangles
    if len(points) != 6:
        raise ValueError("Too many or too few points. Need exactly 6.")

    tris = []

    n = len(points)

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

        while len(L1s) and len(L2s):
            if L1s[0] == L2s[0]: return 1

            if L1s[0] < L2s[0]:
                L1s.pop(0)
            else:
                L2s.pop(0)

        return 0

    i = 0
    j = 1
    while overlap(tris[i][2], tris[j][2]):
        i += 1
        if i == j:
            j += 1
            i = 0

    return [tris[i], tris[j]]

def multipleTrials(n=100, debug=0, debuginner=0, debugfail=0):
    successes = 0
    for i in range(n):
        points = generatePoints(2,3,scramble=0, debug=debuginner)
        pcopy = points[:]
        shuffle(pcopy)
        tris = getTwoTriangles(pcopy, debug=debuginner)
        if debug: print(points,'\n\n',tris)
        if sorted(tris[0][2]) == sorted(points[:3]) or sorted(tris[0][2]) == sorted(points[-3:]):
            successes += 1
        else:
            if debugfail:
                print("Points:",points)
                getTwoTriangles(pcopy, debug=1)

                debugGraph(points, tris)
                
    print("Out of %d trials, there were %d successes and %d failures. That's a %s%% success rate." % (n, successes, n-successes, round(successes/n,4)*100))
    return successes

def debugGraph(points, tris):
    import matplotlib.pyplot as plt

    pointsX, pointsY = zip(*points)
    trueTri1X, trueTri1Y = zip(*points[:3])
    trueTri2X, trueTri2Y = zip(*points[-3:])
    badTri1X, badTri1Y = zip(*tris[0][2])
    badTri2X, badTri2Y = zip(*tris[1][2])

    fig = plt.figure(1)
    sub = fig.add_subplot(111)
    sub.plot(pointsX, pointsY, 'go')
    sub.plot(trueTri1X, trueTri1Y, 'b-')
    sub.plot(trueTri2X, trueTri2Y, 'b-')
    sub.plot(badTri1X, badTri1Y, 'r-')
    sub.plot(badTri2X, badTri2Y, 'r-')

    plt.show()

##iV1 = [1,2]
##iV2 = [3,4]
##
###fV1 = [5,11]
###fV2 = [11,25]
##
##fV1 = [0,4]
##fV2 = [2,6]
##
##print(getAffine(iV1,iV2, fV1,fV2))

##points = generatePoints()
##for point in points: print(point)
###print(findPolys(points, 2,3))
##print(getTwoTriangles(points))

multipleTrials(10,debugfail=1)
