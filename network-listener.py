#!/usr/bin/python
import socket
import struct
import fcntl
import subprocess
import sys
import picamera
import RPi.GPIO
import time

#multi-socket UDP connection
MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind(('', MCAST_PORT))
mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)

sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
    s.fileno(),
    0x8915, # SIOCGIFADDR
    struct.pack('256s', ifname[:15])
    )[20:24])


#which PiCam is this?
id = 1

#create an options file, this file should contain the parameters for the camera
#optionfile = open('/media/Uploaded_Pictures/options.cfg','r')
#options = optionfile.readline()
#optionfile.close()
#he had a simple append below; we're using a different library
#print "options: " + options

camera = picamera.PiCamera()
#camera.exposure_compensation = -15
camera.shutter_speed = 400

while True:
    data = sock.recv(10240)
    data = data.strip()
    if data == "reboot":
        print "rebooting..."
        cmd = 'sudo reboot'
        pid = subprocess.call(cmd, shell=True)
    elif data == "start":
        #with picamera.PiCamera() as camera:
        #Disabling the LED currently not working - bad import? GPIO failure
        #camera.led = False
        camera.resolution = (1920, 1080)
        #start video
        #first erase any existing files
        cmd = 'sudo rm -f temp.h264 temp.mp4'
        pid = subprocess.call(cmd, shell=True)
        #testing network save - unique filenames needed
        camera.start_recording('temp.h264')
        #maximum of 60 seconds to be captured
        #this method allows for error capture - do not use TIME here
        camera.wait_recording(30)
        camera.stop_recording()
    elif data == "transfer":
        #process the file to be useable
        cmd = 'MP4Box -add temp.h264 temp.mp4'
        pid = subprocess.call(cmd, shell=True)
        #copy it to /media/Uploaded_Pictures/
        #cmd = 'sudo cp temp.mp4 /media/Uploaded_Pictures/camera'+str(id)+'-'+time.strftime("M%m-D%d-Y%y-H%H-M%M")+'.mp4'
        cmd = 'sudo cp temp.mp4 /media/Uploaded_Pictures/video'+str(id)+'.mp4'
        print cmd
        pid = subprocess.call(cmd, shell=True)
    elif data == "calibrate":
        camera.start_preview()
    elif data == "stopc":
        camera.stop_preview()
    elif data == "stop":
        #Test to see if camera is recording
        #camera.stop_recording()
        cmd=''
        #this still doesn't work.  :-( use a TRY / CATCH
    elif data == "picture":
        #Take a picture and copy it to the shared folder
        camera.capture('/media/Uploaded_Pictures/camera'+str(id)+time.strftime("M%m-D%d-Y%y-H%H-M%M-S%S")+'.jpeg')
    elif data == "close":
        try:
            camera.close()
            pass
        finally:
            #do nothing
            cmd=''
    elif data == "test":
        camera.capture_sequence(['/media/Uploaded_Pictures/image%02d.jpg' % i for i in range(100)])
