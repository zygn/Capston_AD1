import numpy as np

# sec1
s_m = np.round(np.arange(0,20,0.1999470),7)
x_m = np.linspace(0,-8.0,len(s_m))
y_m = [0 for _ in range(len(s_m))]
psi_rad = [np.round(np.pi, 7) for _ in range(len(s_m))]
zero = [ 0 for _ in range(len(s_m))]
vx_mps = [1 for _ in range(len(s_m))]

# sec2 

s_m2 = np.round(np.arange(20,40,0.1999470),7)
x_m2 = [-8.5 for _ in range(len(s_m2))]
y_m2 = np.linspace(0, 8.5000000,len(s_m2))
psi_rad2 = [np.round(np.pi/2, 7) for _ in range(len(s_m2))]
zero2 = [ 0 for _ in range(len(s_m2))]
vx_mps2 = [1 for _ in range(len(s_m2))]

# sec3

s_m3 = np.round(np.arange(40,80,0.1999470),7)
x_m3 = np.linspace(-8.0,8.0, len(s_m3))
y_m3 = [9.0 for _ in range(len(s_m3))]
psi_rad3 = [0 for _ in range(len(s_m3))]
zero3 = [ 0 for _ in range(len(s_m3))]
vx_mps3 = [1 for _ in range(len(s_m3))]

# sec4

s_m4 = np.round(np.arange(80,100,0.1999470),7)
x_m4 = [8.0 for _ in range(len(s_m4))]
y_m4 = np.linspace(8.5000000,0,len(s_m4))
psi_rad4 = [np.round(-np.pi/2, 7) for _ in range(len(s_m4))]
zero4 = [ 0 for _ in range(len(s_m4))]
vx_mps4 = [1 for _ in range(len(s_m4))]

# # sec5

# s_m5 = np.round(np.arange(100,120,0.1999470),7)
# x_m5 = np.linspace(8.0,0,len(s_m5))
# y_m5 = [0 for _ in range(len(s_m5))]
# psi_rad5 = [np.round(np.pi, 7) for _ in range(len(s_m5))]
# zero5 = [ 0 for _ in range(len(s_m5))]
# vx_mps5 = [1 for _ in range(len(s_m5))]

with open('./new_round/wps.csv','a') as file:
    #sec 1
    for i in range(len(s_m)):
        file.write(f"{s_m[i]}; {x_m[i]}; {y_m[i]}; {psi_rad[i]}; {zero[i]}; {vx_mps[i]}; {zero[i]}\n")
    #sec 2
    for j in range(len(s_m2)):
        file.write(f"{s_m2[j]}; {x_m2[j]}; {y_m2[j]}; {psi_rad2[j]}; {zero2[j]}; {vx_mps2[j]}; {zero2[j]}\n")
    #sec 3
    for k in range(len(s_m3)):
        file.write(f"{s_m3[k]}; {x_m3[k]}; {y_m3[k]}; {psi_rad3[k]}; {zero3[k]}; {vx_mps3[k]}; {zero3[k]}\n")
    # #sec 4
    # for x in range(len(s_m4)):
    #     file.write(f"{s_m4[x]}; {x_m4[x]}; {y_m4[x]}; {psi_rad4[x]}; {zero4[x]}; {vx_mps4[x]}; {zero4[x]}\n")
    #sec 5
    # for y in range(len(s_m5)):
    #     file.write(f"{s_m5[y]}; {x_m5[y]}; {y_m5[y]}; {psi_rad5[y]}; {zero5[y]}; {vx_mps5[y]}; {zero5[y]}\n")

