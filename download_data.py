# reference: https://python-pytube.readthedocs.io/en/latest/user/quickstart.html
#            https://kkroening.github.io/ffmpeg-python/
#            https://pysox.readthedocs.io/en/latest/example.html
import pandas as pd
from pytube import YouTube
import ffmpeg
import sox


def resample_audio(in_path, out_path, sample_rate):
    tfm = sox.Transformer()
    tfm.convert(samplerate=sample_rate)
    tfm.build(in_path, out_path)


def trim_audio(in_path, out_path, start_time, end_time):
    tfm = sox.Transformer()
    tfm.trim(start_time, end_time)
    tfm.build(in_path, out_path)


def download_video(yt_id, start_time, end_time, path_vid, path_aud, path_trim):
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
            op_vid, err = ffmpeg.output(inp, path_vid+'vid_'+str(vid_num+1)+'.mp4').run()
            op_aud, err = ffmpeg.output(inp.audio, path_aud+'vid_'+str(vid_num+1)+'.wav').run()
            resample_audio(path_aud+'vid_'+str(vid_num+1)+'.wav',
                           path_aud+'vid_'+str(vid_num+1)+'.wav', 22100)
            trim_audio(path_aud+'vid_'+str(vid_num+1)+'.wav',
                       path_trim+'vid_'+str(vid_num+1)+'.wav', 5.8, 10)
            vid_num += 1
        except:
            print("Connection Error")  # to handle exception


def main():
    audioset_path = 'data/balanced_train_segments.csv'  # path to the list of YouTube videos
    audioset = pd.read_csv(audioset_path, quotechar='"',
                           skipinitialspace=True, skiprows=2)
    num_videos = 1
    download_video(audioset.iloc[0:num_videos, 0], audioset.iloc[0:num_videos, 1],
                   audioset.iloc[0:num_videos, 2], 'data/train/', 'data/train/full_aud/', 'data/train/trim_aud/')


if __name__ == '__main__':
    main()
