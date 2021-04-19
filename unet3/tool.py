import os
import numpy as np

def get_every(ppath):
  dir1=[]
  for root, dirs, files in os.walk(ppath, topdown=False):
    for name in files:
      dir1.append(os.path.join(ppath,os.path.join(ppath, name)))
      # print(str(name))
  return dir1

def get_files(path):

  # 判断路径是否存在,如果不存在，函数直接结束
  if not os.path.exists(path):
    print('路径不存在')
    return
  # 判断路径是否为文件夹
  if not os.path.isdir(path):
    print('路径是一个文件')
    return
  # 这时候，路径是一个文件夹
  # 获取文件夹中文件或文件夹的名称
  file_list = os.listdir(path)
  dirr=[]
  temp=[[None],[None]]
  j=0
  # print(file_list)
  # 遍历文件夹
  for filename in file_list:
    # 拼接路径，获取每个次级目录下的文件路径
    subpath = os.path.join(path,filename)

    if os.path.isdir(subpath):
      # print(str(subpath))
      temp[j]=get_every(subpath)
      j+=1

  # 分配训练集和验证集的比例
  judge = int(len(temp[0])*0.8)

  with open("object.csv", "w") as f:
      
    for i in range(judge):
      f.write(str(temp[0][i])+","+str(temp[1][i])+"\n")

  with open("vale.csv", "w") as f:
      
    for i in range(judge,len(temp[0])):
      f.write(str(temp[0][i])+","+str(temp[1][i])+"\n")    
        


  print(len(temp))

  

if __name__ == '__main__':
    get_files('E:\BaiduNetdiskDownload\carvana')
    