# reference: https://python-pytube.readthedocs.io/en/latest/user/quickstart.html
#            https://kkroening.github.io/ffmpeg-python/
import pandas as pd
from pytube import YouTube
import ffmpeg


def download_video(yt_id, start_time, end_time, save_path):
    link = ["https://www.youtube.com/watch?v="+id for id in yt_id]
    vid_num = 0
    for i in range(len(link)):
        try:
            # downloading 360p video from YouTube
            yt = YouTube(link[i])
            stream = yt.streams.filter(file_extension='mp4', res='360p').first()
            stream.download('data/temp', filename='tmp')
            # triming the video to specified start time and end time
            inp = ffmpeg.input('data/temp/tmp.mp4', ss=start_time[i], t=end_time[i]-start_time[i])
            op, err = ffmpeg.output(inp, save_path+'vid_'+str(vid_num+1)+'.mp4').run()
            vid_num += 1
        except:
            print("Connection Error")  # to handle exception


def main():
    audioset_path = 'data/balanced_train_segments.csv'  # path to the list of YouTube videos
    audioset = pd.read_csv(audioset_path, quotechar='"',
                           skipinitialspace=True, skiprows=2)
    num_videos = 50
    download_video(audioset.iloc[0:num_videos, 0], audioset.iloc[0:num_videos, 1],
                   audioset.iloc[0:num_videos, 2], 'data/train/')


if __name__ == '__main__':
    main()
