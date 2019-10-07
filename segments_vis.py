import cv2
import numpy as np
import matplotlib.pyplot as plt

def display_frames(axes, frames):
  for i,ax in enumerate(axes):
    print(len(frames), len(axes))
    ax.imshow(frames[i*(len(frames)//len(axes))])
    ax.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
       

def display_strip(ax, labels, name, val):
  # print(val)
  # print(np.expand_dims(val,axis=0))
  ax.pcolor(np.expand_dims(val,axis=0), cmap='tab20',vmin=0,vmax=len(labels))
  # ax.pcolor([[1,2]], cmap='tab20',vmin=0,vmax=len(labels))
  # ax.set_ylabel(name)
  ax.set_ylabel(name,rotation=90, fontsize=16) #rotation=90, , ha='center'
  # ax.set_aspect('equal')
  # ax.tick_params(width=0.5)
  ax.set_yticks([], [])
  ax.set_xticks([], [])
       

def plot(labels, demo_frames, sample_gt, classify_gaze, classify_nogaze, num_frames=15):
  import matplotlib.gridspec as gridspec

  fig = plt.figure(figsize=(num_frames,4),dpi=300) #6
  gs = gridspec.GridSpec(4, num_frames, height_ratios=[10,1,1,1]) #8
 
  # Display Demo
  # for i in range(0,len(demo_frames),len(demo_frames)/num_frames):
  #   print(i)
  display_frames([plt.subplot(gs[0,i]) for i in range(num_frames)], demo_frames)
  display_strip(plt.subplot(gs[1,:]), labels, 'GT', sample_gt)
  display_strip(plt.subplot(gs[2,:]), labels, 'NL', classify_nogaze)
  display_strip(plt.subplot(gs[3,:]), labels, 'NL+\ngaze', classify_gaze)

  plt.subplots_adjust(wspace=0, hspace=0.3)
  plt.savefig('seg_vis.png')
  # plt.show(block=False)
  # plt.close()

labels = [0,1,2,3,4,5]
sample_gt = []
demo_frames = []
classify_gaze = []
classify_nogaze = []
gaze_file = open('titan/out_gaze_yes.txt','r')
no_gaze_file = open('titan/out_gaze_no.txt','r')
gaze_data = gaze_file.read()
gaze_data = gaze_data.split('\n')
no_gaze_data = no_gaze_file.read()
no_gaze_data = no_gaze_data.split('\n')
# print(data)
for d in gaze_data:
  if d!= '':
    d = d.split()
    if 'KT2_4_' in d[0]:
      sample_gt.append(int(d[2]))
      demo_frame = cv2.imread('data/ut-lfd/pouring/VD_images/'+d[0])
      rgb_img = cv2.cvtColor(demo_frame, cv2.COLOR_BGR2RGB)
      rgb_img = cv2.resize(rgb_img, (700,500))
      demo_frames.append(rgb_img)
      classify_gaze.append(int(d[1]))

for d in no_gaze_data:
  if d!= '':
    d = d.split()
    if 'KT2_4_' in d[0]:
      classify_nogaze.append(int(d[1]))

demo = []
import math
for i in range(len(demo_frames)):
  skip = math.floor(len(demo_frames)/15)
  if i%skip==0:
    demo.append(demo_frames[i])
# print(len(demo_frames))
plot(labels, demo, sample_gt, classify_gaze, classify_nogaze)

# TODO: images bigger