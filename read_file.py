file = open("/home/saplab/albert_vi/output/data.txt",'r')


line = file.read().replace("\n\n", "")

file.close()


print(line)
