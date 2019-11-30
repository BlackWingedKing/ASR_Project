import cv2
import numpy as np
import pylab
def cmap_im(cmap, im, lo = None, hi = None):
	return np.uint8(255*apply_cmap(im, cmap, lo, hi))
def apply_cmap(im, cmap = pylab.cm.jet, lo = None, hi = None):
	return cmap(clip_rescale(im, lo, hi).flatten()).reshape(im.shape[:2] + (-1,))[:, :, :3]
def clip_rescale(x, lo = None, hi = None):
	if lo is None:
		lo = np.min(x)
	if hi is None:
		hi = np.max(x)
	return np.clip((x - lo)/(hi - lo), 0., 1.)

def heatmap(frames, cam, lo_frac = 0.5, adapt = True, max_val = 35):
	""" Set heatmap threshold adaptively, to deal with large variation in possible input videos. """
	frames = np.asarray(frames)
	max_prob = 0.35
	if adapt:
		max_val = np.percentile(cam, 97)

	same = np.max(cam) - np.min(cam) <= 0.001
	if same:
		return frames

	outs = []
	print(frames.shape)
	for i in range(frames.shape[0]):
		lo = lo_frac * max_val
		hi = max_val + 0.001
		im = frames[i]
		f = cam.shape[0] * float(i) / frames.shape[0]
		l = int(f)
		r = min(1 + l, cam.shape[0]-1)
		p = f - l
		frame_cam = ((1-p) * cam[l]) + (p * cam[r])
		print(frame_cam.shape)
		frame_cam = cv2.resize(frame_cam, (im.shape[1], im.shape[0]))
		print(frame_cam.shape)
		# frame_cam = ig.scale(frame_cam, im.shape[:2], 1)
		#vis = ut.cmap_im(pylab.cm.hot, np.minimum(frame_cam, hi), lo = lo, hi = hi)
		vis = cmap_im(pylab.cm.jet, frame_cam, lo = lo, hi = hi)
		print(vis)
		#p = np.clip((frame_cam - lo)/float(hi - lo), 0, 1.)
		p = np.clip((frame_cam - lo)/float(hi - lo), 0, max_prob)
		p = p[..., None]
		im = np.array(im, 'd')
		vis = np.array(vis, 'd')
		# print(im,vis)
		print(p,"p")
		outs.append(np.uint8(im*(1-p) + vis*p))
	return np.array(outs)
def findcam2(video_frames,cam):
	ims = video_frames[0]
	# print(ims,cam)
	cam = cam[0]
	print("ims",ims.shape,ims[0][np.newaxis].shape)
	for i in range(cam.shape[0]):
		vis = heatmap(ims[i*(125//16 + 1)][np.newaxis], cam[i][np.newaxis], adapt = True)
		print(vis,vis.shape)
		cv2.imwrite('cam'+str(i)+'.jpg', vis[0])
	# cam shape batch,time(16),7*7
	# video_frames batch, time(sample first 16),h,w.c
def findcam(video_frames,cam):

	print(video_frames.shape,cam.shape)
	t = video_frames.shape[1]
	for i in range(cam.shape[0]):
		for j in range(cam.shape[1]):
			height, width, _ = video_frames[i][j*(t//16 + 1)].shape
			cam[i][j] = cam[i][j] - np.min(cam[i][j])
			cam_img = cam[i][j] / np.max(cam[i][j])
			cam_img = np.uint8(255 * cam_img)
			CAM = cv2.resize(cam_img, (width, height))
			heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
			heatmap =np.clip(heatmap,a_min=0,a_max = 255)
			result = heatmap * 0.3 + video_frames[i][j*(t//16 + 1)] * 0.7
			cv2.imwrite('cam'+str(i)+str(j)+'.jpg', result)