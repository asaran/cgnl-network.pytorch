import os
# import rosbag
import json
import numpy as np
import cv2
import soundfile as sf
import pickle as pkl
class LfdData():
    def __init__(self, exp, demo_type,skip=1):
        self.demo_type = demo_type
        if demo_type=='video':
            img_dir = 'VD_images' #_human
            # self.feat_file = '../data/video_vggish_feats/'+user+'/'+self.task+'/vggish_env_py2.pkl'
        elif demo_type=='kt':
            img_dir = 'KD_images' #_human
            # self.feat_file = '../data/kt_vggish_feats/'+user+'/'+self.task+'/vggish_env_kt.pkl'
        base_dir = '/media/akanksha/Seagate Portable Drive/audio_study/segmentation_data/'
        self.dest_dir = base_dir + exp + '/' + img_dir#'../data/box/'
        self.skip = skip
        # TODO: only users who are talking
        self.users = ['user2', 'user4', 'user7', 'user8', 'user10','user14','user16','user18','user20'] 
        self.img_topic = '/camera/right/qhd/image_color_rect/compressed'
        self.data_dir = '/media/akanksha/Seagate Portable Drive/audio_study/kinesthetic/'
        self.data_dir_ = '/media/akanksha/Seagate\ Portable\ Drive/audio_study/kinesthetic/'
        self.gt_path = '/home/akanksha/Documents/audio_lfd/experiments/subtask_detection/data/mturk-annotation/'
        self.task = exp
        self.img_counts = {}
        t_file = base_dir+'/'+exp+'/train_'+self.task+'_'+self.demo_type+'.list'
        v_file = base_dir+'/'+exp+'/val_'+self.task+'_'+self.demo_type+'.list'
        self.f_train = open(t_file,'w')
        self.f_val = open(v_file,'w')

    def generate(self):
        # iterate through users
        for user in self.users:
            # get gt annotation of labels
            print(user)
            gt_path = os.path.join(self.gt_path,user,self.task,self.demo_type)
            for f in os.listdir(gt_path):
                if f.endswith('.json'):
                    gt_file = os.path.join(gt_path,f)
                    with open(gt_file, "r") as read_file:
                        gt = json.load(read_file)
                    break
            # print(gt_file)
            # assert(gt_file.endswith('.json'))
            

            # open bag file
            # bagfiles = []
            bag_path = os.path.join(self.data_dir,user,self.task,self.demo_type) 
            # for dirname, dirs, files in os.walk(bag_path):
            #     for filename in files:
            #         fn,ex = os.path.splitext(filename)
            #         if ex == '.bag':
            #             bagfiles.append(filename)
            # bagfiles.sort(reverse=True)
            # bag = os.path.join(bag_path,bagfiles[0])

            # save every image
            img_base = self.dest_dir + '/' + user +'_'
            img_names = [p for p in os.listdir(self.dest_dir) if p.endswith('.jpg') and user+'_' in p]
            img_names.sort()
            #video_command = 'ffmpeg -framerate 10 -i '+img_base+'_%04d.jpg \
            #     -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p -filter_complex' + bag_path
            # bag_audio = rosbag.Bag(bag) 
            # for idx, data in enumerate(bag_audio.read_messages(topics=[self.img_topic])):
            #     topic, msg, t = data
            for img_name in img_names:
                flag = True
                img_path = os.path.join(img_base,img_name)
                idx = int(img_name.split('_')[-1][:-4])
                # print(idx)

                # sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
                if idx%self.skip==0:
                    # img = msg.data
                    # np_arr = np.fromstring(msg.data, np.uint8)
                    # image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    # cv2.imwrite(img_base+str(idx)+'.jpg',image_np)
                                   

                    # compute label for image
                    curr_time = float(idx)/10.0 #secs #assume 10fps
                    for action in gt:
                        assert(len(gt[action])%3==0)
                        for t in range(int(len(gt[action])/3)):
                            start_time = gt[action][t*3]
                            end_time = gt[action][t*3+1]
                            if curr_time<=end_time and curr_time>=start_time:
                                label = str(int(action))
                                # print(idx, label, flag)
                                flag = False
                                # print('****Action found!!!!!!', str(idx))
                                break
                            # else:
                            # print('no action for this frame, skipping...',str(idx))
                            # flag = True
                            
                            
                    if flag:
                        flag = False
                        # print('continue...')
                        # TODO: add a noise/none label
                        continue

                    # load corresponding vggish features for audio around image (0.96 secs each - 128 dim)
                    audio_file = bag_path+'/env.wav'
                    audio,sr = sf.read(audio_file)
                    # get total length of audio file
                    audio_len = len(audio)/float(sr)
                    
                    if self.demo_type=='video':
                        feat_file = '../data/video_vggish_feats/'+user+'/'+self.task+'/vggish_env_py2.pkl'
                    elif self.demo_type=='kt':
                        feat_file = '../data/kt_vggish_feats/'+user+'/'+self.task+'/vggish_env_kt.pkl'
                    vggish_feats = pkl.load(open(feat_file,'rb')) # which 0.96 sec time frame to pick for 128 dim feat?
                    # print(type(vggish_feats))
                    audio_fps = vggish_feats.shape[0]/float(audio_len)
                    feats_idx = int(audio_fps*curr_time)
                    curr_feats = vggish_feats[feats_idx]

                    
                    line = img_name+' '+label+' '+' '.join(map(str,curr_feats))+'\n'
                    # write to val.list (users 18-20)
                    # print(label)
                    if user=='user18' or user=='user20':
                        # print(user)
                        # exit(1)
                        self.f_val.write(line)

                    # write to train.list
                    else:
                        self.f_train.write(line)

                    
            # bag_audio.close()
            self.img_counts[user] = idx-1
        
        self.f_train.close()
        self.f_val.close()
            

if __name__ == "__main__":
    data = LfdData('box', 'kt', skip=2)
    data.generate()
