import socket
import sys
import time

print 'Network Controller Started'
#pass along command line arguments such as start, stop, reboot, or name of photo series
n = sys.stdin.readline()
n = n.strip('\n')

#setup networking configuration (UDP broadcast)
MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007

#network code; broadcasts message "n"
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
sock.sendto(n, (MCAST_GRP, MCAST_PORT))
