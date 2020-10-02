import os
import rosbag
import json
import numpy as np
import cv2

class LfdData():
    def __init__(self, exp, demo_type,skip=1):
        self.demo_type = demo_type
        if demo_type=='video':
            img_dir = 'VD_images'
        elif demo_type=='kt':
            img_dir = 'KD_images'
        self.dest_dir = '/media/akanksha/Seagate Portable Drive/audio_study/segmentation_data/'+ exp + '/' + img_dir#'../data/box/'
        self.skip = skip
        self.users = ['user2', 'user4', 'user7', 'user8', 'user10','user14','user16','user18','user20'] 
        self.img_topic = '/camera/right/qhd/image_color_rect/compressed'
        self.data_dir = '/media/akanksha/Seagate Portable Drive/audio_study/kinesthetic/'
        self.data_dir_ = '/media/akanksha/Seagate\ Portable\ Drive/audio_study/kinesthetic/'
        self.gt_path = '/home/akanksha/Documents/audio_lfd/experiments/subtask detection/mturk-annotation/mturk-annotation/'
        self.task = exp
        self.img_counts = {}

    def generate(self):
        # iterate through users
        for user in self.users:
            # get gt annotation of labels
            print(user)
            gt_path = os.path.join(self.gt_path,user,self.task,self.demo_type)
            for f in os.listdir(gt_path):
                if f.endswith('.json'):
                    gt_file = f
                    break
            # print(gt_file)
            # assert(gt_file.endswith('.json'))
            

            # open bag file
            bagfiles = []
            bag_path = os.path.join(self.data_dir,user,self.task,self.demo_type) 
            for dirname, dirs, files in os.walk(bag_path):
                for filename in files:
                    fn,ex = os.path.splitext(filename)
                    if ex == '.bag':
                        bagfiles.append(filename)
            bagfiles.sort(reverse=True)
            bag = os.path.join(bag_path,bagfiles[0])

            # video_path = os.path.join(self.data_dir_,user,self.task,self.demo_type) 
            # runner = 'rosrun bag_tools make_video.py '
            # runner += self.img_topic + ' ' + bag + ' --output ' + os.path.join(video_path,'right_view.mp4')
            # print(runner+'\n')
            # os.system(runner)

            # save every image
            img_base = self.dest_dir + '/' + user +'_'
            #video_command = 'ffmpeg -framerate 10 -i '+img_base+'_%04d.jpg \
            #     -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -filter_complex' + bag_path
            bag_audio = rosbag.Bag(bag) 
            for idx, data in enumerate(bag_audio.read_messages(topics=[self.img_topic])):
                topic, msg, t = data
                # sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
                if idx%self.skip==0:
                    img = msg.data
                    np_arr = np.fromstring(msg.data, np.uint8)
                    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    cv2.imwrite(img_base+str(idx)+'.jpg',image_np)
            bag_audio.close()
            self.img_counts[user] = idx-1

            # compute label for image

            # compute corresponding vggish features for audio around image

            # write to train.list

            # write to val.list (users 18-20)

if __name__ == "__main__":
    data = LfdData('cutting', 'video',skip=2)
    data.generate()
