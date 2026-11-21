import struct
w=struct.unpack("<12q",open("/dev/shm/e3_dsv4pro_peer","rb").read(96))
print(dict(published=w[0],completed=w[1],timeouts=w[2],rows=w[3],tokens=w[4],wait_s=round(w[5]/1e9,2),waits=w[6]))
