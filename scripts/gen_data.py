from numpy import ones,vstack
from numpy.linalg import lstsq
import cv2
import ast 
from bisect import bisect_left
import os
import gzip

class LfdData():
	def __init__(self, exp):
		self.data_dir = '../data/ut-lfd/'+ exp + '/' #'../data/pouring/experts/KT6/5fyyvco/segments/6/'
		self.visualize = False
		self.users = []
		self.user_dir = ''
		self.gp = {}
		self.vid2ts = {}
		self.all_vts = []
		self.model = []
		self.vid_start = {}
		self.vid_end = {}
		# self.reach, self.grasp, self.trans, self.pour, self.ret, self.vid_rel = {}, {}, {}, {}, {}, {}
		self.seg_times, self.segs = {}, {}
		self.labels = { 'reach': 1,
						'grasp': 2,
						'transport': 3,
						'pour': 4,
						'release': 5,
						'return': 6}
		self.order = {'KT1':'kv','KT2':'kv','KT3':'vk','KT4':'vk','KT5':'kv','KT6':'kv','KT7':'vk','KT8':'vk','KT9':'kv','KT10':'kv',\
		'KT11':'vk','KT12':'vk','KT13':'kv','KT14':'kv','KT15':'vk','KT16':'vk','KT17':'kv','KT18':'vk','KT19':'vk','KT20':'vk'}

		f = open(self.data_dir +"images.txt",'w')
		f2 = open(self.data_dir +"image_class_labels.txt","w")
		f3 = open(self.data_dir +"image_gaze.txt","w")
		f4 = open(self.data_dir +"train.txt","w")
		f5 = open(self.data_dir +"val.txt","w")
		f.close()
		f2.close()
		f3.close()
		f4.close()
		f5.close()
		self.img_count = 0

	def read_json(self, my_dir):
		data_file = my_dir+"livedata.json.gz"
		with gzip.open(data_file, "rb") as f:
			data=f.readlines()

		for r in range(len(data)):
			row = data[r]
			data[r] = ast.literal_eval(row.strip('\n'))

		self.vid2ts = {}     # dictionary mapping video time to time stamps in json
		right_eye_pd, left_eye_pd, self.gp = {}, {}, {} # dicts mapping ts to pupil diameter and gaze points (2D) for both eyes

		for d in data:
			if 'vts' in d and d['s']==0:
				if d['vts'] == 0:
					self.vid2ts[d['vts']] = d['ts']
				else:
					self.vid2ts[d['vts']] = d['ts']

			# TODO: if multiple detections for same time stamp?
			if 'pd' in d and d['s']==0 and d['eye']=='right':
				right_eye_pd[d['ts']] = d['pd']
			if 'pd' in d and d['s']==0 and d['eye']=='left':
				left_eye_pd[d['ts']] = d['pd']

			if 'gp' in d and d['s']==0 :
				self.gp[d['ts']] = d['gp']   #list of 2 coordinates
		print('read json')

		# map vts to ts
		self.all_vts = sorted(self.vid2ts.keys())
		a = self.all_vts[0]
		self.model = []
		for i in range(1,len(self.all_vts)):
			points = [(a,self.vid2ts[a]),(self.all_vts[i],self.vid2ts[self.all_vts[i]])]
			x_coords, y_coords = zip(*points)
			A = vstack([x_coords, ones(len(x_coords))]).T
			m, c = lstsq(A, y_coords)[0]
			# print("Line Solution is ts = {m}vts + {c}".format(m=m,c=c))
			self.model.append((m,c))


	
	def takeClosest(self, myList, myNumber):
		"""
		Assumes myList is sorted. Returns closest value to myNumber.
		If two numbers are equally close, return the smallest number.
		"""
		pos = bisect_left(myList, myNumber)
		if pos == 0:
			return myList[0]
		if pos == len(myList):
			return myList[-1]
		before = myList[pos - 1]
		after = myList[pos]
		if after - myNumber < myNumber - before:
		   return after
		else:
		   return before


	def gaze_for_imgs(self, seg, user, trial, skip):
		user_dir = self.data_dir + 'videos/' + user + '/' + seg + '/segments/' + str(trial) + '/'
		self.read_json(user_dir)
		vidcap = cv2.VideoCapture(user_dir+'fullstream.mp4')
		fps = vidcap.get(cv2.CAP_PROP_FPS)
		frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
		end_time = frame_count/fps
		# print fps 	#25 fps
		success, img = vidcap.read()

		if self.visualize:
			fourcc = cv2.VideoWriter_fourcc(*'XVID')
			video = cv2.VideoWriter(user_dir+'gaze_overlayed.avi',fourcc,fps,(1920,1080))

		all_ts = sorted(self.gp.keys())
		count = 0
		imgs = []       # list of image frames
		frame2ts = []   # corresponding list of video time stamp values in microseconds

		# print(type(fps),type(self.vid_start[user]))
		start = abs(fps*self.vid_start[user])
		if self.vid_end[user]=='end':
			end = abs(fps*end_time)
		else:
			end = abs(fps*self.vid_end[user])

		# open files for image names and labels
		f = open(self.data_dir +"images.txt",'a')
		f2 = open(self.data_dir +"image_class_labels.txt","a")
		f3 = open(self.data_dir +"image_gaze.txt","a")
		f4 = open(self.data_dir + "train.txt", 'a')
		f5 = open(self.data_dir + "val.txt", 'a')

		while success:	
			# print(count)
			if count>=start and count<=end:
				frame_ts = int((count/fps)*1000000)
				frame2ts.append(frame_ts)

				less = [a for a in self.all_vts if a<=frame_ts]
				idx = len(less)-1

				if idx<len(self.model):
					m,c = self.model[idx]
				else:
					m,c = self.model[len(self.model)-1]
				ts = m*frame_ts + c

				tracker_ts = self.takeClosest(all_ts,ts)

				gaze = self.gp[tracker_ts]
				gaze_coords = (int(gaze[0]*1920), int(gaze[1]*1080))

				if self.visualize:
					img_hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
					hsv = img_hsv[gaze_coords[1]][gaze_coords[0]]
					font = cv2.FONT_HERSHEY_SIMPLEX
					color_name, color_value = get_color_name(hsv)
					
					if(color_name!=''):
					# 	print(color_name)
						cv2.putText(img, color_name, (1430, 250), font, 1.8, color_value, 5, cv2.LINE_AA)

					# print(hsv)
					cv2.putText(img, str(hsv), (230, 250), font, 1.8, (255, 255, 0), 5, cv2.LINE_AA)

					cv2.circle(img,gaze_coords, 25, (255,255,0), 3)
					video.write(img)

				if(count%skip==0):
					# write images
					self.img_count+=1
					img_name = user+'_'+str(trial)+'_'+str(count)+'.jpg'
					cv2.imwrite(self.data_dir+'/VD_images/'+img_name, img)
					f.write(str(self.img_count)+' '+img_name+'\n')
					seg = self.find_segment(user, count, fps, end_time)
					f2.write(str(self.img_count)+' '+str(self.labels[seg])+'\n')
					# TODO: write normalized gaze coordinates
					f3.write(str(self.img_count)+' '+str(gaze[0])+' '+str(gaze[1])+'\n')

					if(user=='KT19' or user=='KT20'):
						f5.write(img_name+' '+str(self.labels[seg])+'\n')
					else:
						f4.write(img_name+' '+str(self.labels[seg])+'\n')

			count += 1
			success, img = vidcap.read()

		if self.visualize:
			cv2.destroyAllWindows()
			video.release()

		f.close()
		f2.close()
		f3.close()
		f4.close()
		f5.close()
	
	def read_annotation_file(self):
		file_path = self.data_dir+'video_kf.txt'
		file = open(file_path, 'r') 

		for line in file:
			entries = line.strip('\n').split(' ')
			user = entries[0]
			self.users.append(user)
			self.vid_start[user] = float(entries[1])
			if entries[-1]=='end':
				self.vid_end[user] = entries[-1]  
			else:
				# print('last entry:', entries[-1])
				self.vid_end[user] = float(entries[-1])

			self.seg_times[user] = [float(e) if e!='end' else e for e in entries[2:-1] ]
			self.segs[user] = {}
			self.segs[user][float(entries[2])] = 'reach' 
			self.segs[user][float(entries[8])] = 'reach'
			self.segs[user][float(entries[3])] = 'grasp'
			self.segs[user][float(entries[9])] = 'grasp'
			self.segs[user][float(entries[4])] = 'transport'
			self.segs[user][float(entries[10])] = 'transport'
			self.segs[user][float(entries[5])] = 'pour'
			self.segs[user][float(entries[11])] = 'pour'
			self.segs[user][float(entries[6])] = 'return'
			self.segs[user][float(entries[12])] = 'return'
			self.segs[user][float(entries[7])] = 'release' 
			if entries[13]!='end':
				self.segs[user][float(entries[13])] = 'release' 
			else:
				self.segs[user][entries[13]] = 'release' 

	def find_segment(self, user, count, fps, end_time):
		# print(self.segs[user])
		# print(user)
		for i in range(len(self.seg_times[user])-1):
			s = self.seg_times[user][i]
			s_next = self.seg_times[user][i+1]
			t = s
			if s_next=='end':
				t_next = end_time
			else:
				t_next = s_next
			# print('t: ',str(t))
			if count>=t*fps and count<=t_next*fps:
				return self.segs[user][s]
		return self.segs[user][s]

	def create_imgs(self):
		self.read_annotation_file()
		# print(self.users)
		for user in self.users:
			print(user)
			exps = self.order[user]
			demo_type = 'v'
			if demo_type == exps[0]:
				trial = 1
			else:
				trial = 4

			user_dir = self.data_dir + 'videos/' + user + '/'
			d = os.listdir(user_dir)
			assert(len(d)==1)
			seg = d[0]
			# print(seg)
			# trial = 4
			self.gaze_for_imgs(seg, user, trial, 2)


	def get_color_name(self,hsv):

		color_ranges = {
			'red':   [[161,140,70],[184,255,255]],
			'green': [[36,64,28],[110,155,220]], #[[36,64,28],[70,155,220]]
			'yellow': [[0,90,100],[32,180,180]],
			'blue': [[94,111,34],[118,165,136]],
			'black': [[0,0,0],[180,255,40]],
			'white': [[0,0,170],[180,255,255]]
		}

		color_val = {
			'black': (0,0,0),
			'white': (255,255,255),
			'red': (0,0,255),
			'green': (0,255,0),
			'yellow': (0,255,255),
			'blue': (255,0,0),
			'pasta': (0,215,225)
		}

		h,s,v = hsv
		color = ''
		value = None
		for i, (n,r) in enumerate(color_ranges.items()):
			# print(n, r[0][0], r[1][0])
			if h>=r[0][0] and h<=r[1][0]:
				if s>=r[0][1] and s<=r[1][1]:
					if v>=r[0][2] and v<=r[1][2]:
						color = n 
						value = color_val[n]

		pasta_color_range = [[0,30,0],[40,130,100]]
		p = pasta_color_range
		if color=='':
			if h>=p[0][0] and h<=p[1][0]:
				if s>=p[0][1] and s<=p[1][1]:
					if v>=p[0][2] and v<=p[1][2]:
						color = 'pasta'
						value = color_val['pasta']

		return color, value




if __name__ == "__main__":
	data = LfdData('pouring')
	data.create_imgs()
	# data.create_train_val()
