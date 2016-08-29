import socket


TCP_IP = '192.168.1.64'
TCP_PORT = 10000
BUFFER_SIZE = 2
MESSAGE = "*+155|-188#".encode("utf-8")

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
s.send(MESSAGE)
# data = s.recv(BUFFER_SIZE)
data =None
s.close()

print("received data:", data)