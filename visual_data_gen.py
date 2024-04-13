import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from PIL import Image, ImageSequence

data = np.load("/Users/dp/Documents/清华/Filter/data/sol_1.npy")
if os.path.exists('results'):
    shutil.rmtree('results')
os.mkdir('results')

for i in range(data.shape[0]):
    plt.figure()
    plt.imshow(data[i,:,:])
    plt.colorbar()
    plt.clim(-0.5,0.5)
    #plt.yscale('log')
    plt.savefig('results/'+str(i)+'.png')
    plt.close()
png_images = ['results/'+str(i)+'.png' for i in range(data.shape[0])]
# 输出GIF文件名
output_gif = "output.gif"
 
# 打开第一张图片来创建动画
first_image = Image.open(png_images[0])
 
# 创建一个动画的帧列表
frames = [Image.open(img) for img in png_images]
 
# 设置GIF的参数，如延迟等
duration = 0.05  # 每帧的持续时间（秒）
 
# 保存为GIF
first_image.save(output_gif, save_all=True, append_images=frames, duration=duration, loop=0)
shutil.rmtree('results')