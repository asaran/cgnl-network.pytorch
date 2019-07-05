import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_frames(axes, frames):
  for i,ax in enumerate(axes):
    ax.imshow(frames[i*(len(frames)//len(axes))])
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
       

def display_strip(ax, labels, name, val):
  # print(val)
  # print(np.expand_dims(val,axis=0))
  ax.pcolor(np.expand_dims(val,axis=0), cmap='tab20',vmin=0,vmax=len(labels))
  # ax.pcolor([[1,2]], cmap='tab20',vmin=0,vmax=len(labels))
  ax.set_ylabel(name,rotation=90, fontsize=10, ha='center')
  ax.set_aspect('equal')
  # ax.tick_params(width=0.5)
  ax.set_yticks([], [])
  ax.set_xticks([], [])
       

def plot(labels, demo_frames, sample_gt, classify, num_frames=20):
  import matplotlib.gridspec as gridspec

  fig = plt.figure(figsize=(num_frames,3),dpi=300)
  gs = gridspec.GridSpec(8, num_frames)
 
  # Display Demo
  for i in range(0,len(demo_frames),len(demo_frames)/num_frames):
    print(i)
  display_frames([plt.subplot(gs[0:3,i]) for i in range(num_frames)], demo_frames)
  display_strip(plt.subplot(gs[3,:]), labels, 'GT', sample_gt)
  display_strip(plt.subplot(gs[4,:]), labels, 'NL', classify)
  display_strip(plt.subplot(gs[5,:]), labels, 'NL+\ngaze', classify)

  plt.subplots_adjust(wspace=0, hspace=0.2)
  plt.savefig('seg_vis.png')
  # plt.show(block=False)
  # plt.close()

labels = [0,1,2,3,4,5]
sample_gt = []
demo_frames = []
classify = []
data_file = open('data/ut-lfd/pouring/train.list','r')
data = data_file.read()
data = data.split('\n')
# print(data)
for d in data:
  if d!= '':
    d = d.split()
    if 'KT1_4_' in d[0]:
      sample_gt.append(int(d[1]))
      demo_frame = cv2.imread('data/ut-lfd/pouring/VD_images/'+d[0])
      rgb_img = cv2.cvtColor(demo_frame, cv2.COLOR_BGR2RGB)
      demo_frames.append(rgb_img)
      classify.append(int(d[1]))

demo = []
import math
for i in range(len(demo_frames)):
  skip = math.floor(len(demo_frames)/20)
  if i%skip==0:
    demo.append(demo_frames[i])
plot(labels, demo, sample_gt, classify)

# TODO: images bigger