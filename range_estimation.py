import numpy as np
import matplotlib as plt
# txdata:发送端QAM序列（IDFT前）
# rxdata:接收端QAM序列（FFT后）
# rangeE:距离估计值
def range_estimate(txdata,rxdata,N_fft):
    divide_array=np.divide(rxdata,txdata)
    nor_power=abs(np.fft.ifft(divide_array,N_fft,1))
    mean_nor_power=np.mean(nor_power,2)






if __name__ =="__main__":
    txdata=np.ones([4096,1])
    rxdata=np.ones([4096,1])*2
