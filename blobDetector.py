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
from math import sqrt
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

cD = colDis(targetColor)

print("Ready to start!")

#botrow = [0]*xdim
#rigcol = [0]*ydim

##for y in range(ydim-1):
##    for x in range(xdim-1):
##        a = int(image[y][x][0])
##        b = int(image[y+1][x+1][0])
##        c = np.uint8( 128-(a-b)//2 )
##
##        for z in range(3):
##            image[y][x][z] = c

rigcol = []

for y in range(ydim):
    rowmin = min(map(cD,image[y]))
    rigcol.append([gray(255*rowmin/442)])

rc = np.array(rigcol)
rct = np.tile(rc, (1,10,1))

print("Done with rows!")

imageT = np.transpose(image, (1,0,2))
botrow = []

for x in range(xdim):
    colmin = min(map(cD,imageT[x]))
    botrow.append(gray(255*colmin/422))
botrow.extend([gray(0)]*10)

br = np.array([botrow])
brt = np.tile(br, (10,1,1))

print("Done with columns!")

newImage = np.hstack((image,rct))
newImage = np.vstack((newImage,brt))

print("All done!")

plt.imshow(newImage)
plt.show()
