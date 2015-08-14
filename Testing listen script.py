#!/usr/bin/python
import socket
import struct
import fcntl
import subprocess
import sys
import picamera

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

"""
id = get_ip_address('eth0')

ip1, ip2, ip3, ip4 = id.split('.')
#uses the last IP address part as an identifier -- good idea
print 'ID: ' + ip4 
"""

#create an options file, this file should contain the parameters for the camera
#optionfile = open('/media/Uploaded_Pictures/options.cfg','r')
#options = optionfile.readline()
#optionfile.close()
#he had a simple append below; we're using a different library
#print "options: " + options

while True:
    data = sock.recv(10240)
    data = data.strip()
    if data == "reboot":
        print "rebooting..."
        cmd = 'sudo reboot'
        pid = subprocess.call(cmd, shell=True)
    elif data == "start":
    	#start video
    	with picamera.PiCamera() as camera:
            camera.resolution = (1920, 1080)
            #testing network save - unique filenames needed
            camera.start_recording('/media/Uploaded_Pictures/test.h264')
            #maximum of 60 seconds to be captured
            #this method allows for error capture - do not use TIME here
            camera.wait_recording(60)
            camera.stop_recording()
    elif data == "stop":
    	#stop video
    	camera.stop_recording()
    elif data == "auto":
    	#autoexpose
    	#May not be necessary - hardcoded into start for now
    else:
    	"""
    	Re-write all this.  Needs to loop until the stop command or a set number of images
    	(to avoid just filling up memory/storage with a runaway capture sequence
    	No more than 30 seconds
    	
    	We need more than just two options to be parsed here
    	"" "
        print "shooting " + data 
        cmd = 'raspistill -o /tmp/photo.jpg ' + options 
        pid = subprocess.call(cmd, shell=True)
        print "creating directory"
        cmd = 'mkdir /server/3dscan/' + data 
        pid = subprocess.call(cmd, shell=True)
        print "copy image"
        cmd = 'cp /tmp/photo.jpg /server/3dscan/' + data + "/" + data + "_" + ip4 + '.jpg'
        pid = subprocess.call(cmd, shell=True)
        print "photo uploaded"
        """
		print 'error #229a - unspecificied input'