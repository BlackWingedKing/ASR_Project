import cv2
def findcam(video_frames,cam):
	# cam shape batch,time(16),7*7
	# video_frames batch, time(sample first 16),h,w.c
	for i in range(cam.shape[0]):
		for j in range(cam.shape[1]):
			height, width, _ = video_frames[i][j]
			cam[i][j] = cam[i][j] - np.min(cam[i][j])
			cam_img = cam[i][j] / np.max(cam[i][j])
			cam_img = np.uint8(255 * cam_img)
			CAM = cv2.resize(cam_img, (width, height))
			heatmap = cv2.applyColorMap(CAM, cv2.COLORMAP_JET)
			result = heatmap * 0.3 + video_frames[i][j] * 0.7
			cv2.imwrite('cam'+str(i)+str(j)+'.jpg', result)