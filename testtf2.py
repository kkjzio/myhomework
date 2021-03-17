
#!/usr/bin/env python
#!encoding=utf-8
 
import matplotlib
import matplotlib.pyplot as plt
 
#if __name__ == '__main__':
    #fig1=plt.figure(figsize=(8, 10))
#    for i in range(2):
 #       plt.subplot(221+i)

fig,axs=plt.subplots(2,3,figsize=(15,15),sharex=False,sharey=False)        
plt.show()