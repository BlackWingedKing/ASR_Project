# reference: https://python-pytube.readthedocs.io/en/latest/user/quickstart.html
#            https://kkroening.github.io/ffmpeg-python/
#            https://pysox.readthedocs.io/en/latest/example.html
import pandas as pd
from pytube import YouTube
import ffmpeg
import sox
import numpy as np
from joblib import Parallel, delayed
import os


def resample_audio(in_path, out_path, sample_rate):
    tfm = sox.Transformer()
    tfm.convert(samplerate=sample_rate)
    tfm.build(in_path, out_path)


def trim_audio(in_path, out_path, start_time, end_time):
    tfm = sox.Transformer()
    tfm.trim(start_time, end_time)
    tfm.build(in_path, out_path)


def download_video(yt_id, start_time, end_time, fps, aud_sampling, path_vid, path_shift, path_unshift, rng):
    link = ["https://www.youtube.com/watch?v="+id for id in yt_id]
    vid_num = 0
    for i in range(len(link)):
        try:
            # downloading 360p video from YouTube
            yt = YouTube(link[i])
            stream = yt.streams.filter(file_extension='mp4', res='360p').first()
            stream.download('data/temp', filename='tmp')
            # triming the video to specified start time and end time (according to audioset)
            inp = ffmpeg.input('data/temp/tmp.mp4',
                               ss=start_time[i], t=end_time[i]-start_time[i], r=fps)
            filename = path_vid+'vid_'+str(vid_num+1)+'.mp4'
            op_aud, err = ffmpeg.output(inp.audio, 'data/temp/tmp_aud.wav').run()
            resample_audio('data/temp/tmp_aud.wav', 'data/temp/tmp_aud_resampled.wav', aud_sampling)
            aud_resampled = ffmpeg.input('data/temp/tmp_aud_resampled.wav')
            op_vid, err = ffmpeg.output(inp.video, aud_resampled.audio, filename).run()
            # taking randomly sampled 4.2 sec video from original 10 sec video
            start_time = round(2 + rng.rand()*3.8, 2)
            inp_unshift = ffmpeg.input(filename, ss=start_time, t=4.2)
            op_unshift, err = ffmpeg.output(inp_unshift,
                                            path_unshift+'vid_'+str(vid_num+1)+'.mp4').run()
            # shifting audio
            trim_audio('data/temp/tmp_aud_resampled.wav', 'data/temp/tmp.wav', 5.8, 10)
            vid = inp_unshift.video
            aud = ffmpeg.input('data/temp/tmp.wav')
            op_shift, err = ffmpeg.output(vid, aud.audio,
                                          path_shift+'vid_'+str(vid_num+1)+'.mp4').run()
            vid_num += 1
        except:
            print("Connection Error")  # to handle exception


def download_yt_link(vid_num, link, start_time, end_time, fps, aud_sampling, path_vid, path_shift,
                     path_unshift, rng):
    try:
        yt = YouTube(link)
        stream = yt.streams.filter(file_extension='mp4', res='360p').first()
        tmp = 'tmp_'+str(vid_num)
        stream.download('data/temp', filename=tmp)
    except:
        return
    # triming the video to specified start time and end time (according to audioset)
    inp = ffmpeg.input('data/temp/'+tmp+'.mp4',
                       ss=start_time, t=end_time-start_time, r=fps)
    filename = path_vid+'vid_'+str(vid_num)+'.mp4'
    op_aud, err = ffmpeg.output(inp.audio, 'data/temp/'+tmp+'_aud.wav').run()
    resample_audio('data/temp/'+tmp+'_aud.wav', 'data/temp/'+tmp+'_aud_resampled.wav', aud_sampling)
    aud_resampled = ffmpeg.input('data/temp/'+tmp+'_aud_resampled.wav')
    op_vid, err = ffmpeg.output(inp.video, aud_resampled.audio, filename).run()

    # taking randomly sampled 4.2 sec video from original 10 sec video
    start_time = round(2 + rng*3.8, 2)
    inp_unshift = ffmpeg.input(filename, ss=start_time, t=4.2)
    op_unshift, err = ffmpeg.output(inp_unshift,
                                    path_unshift+'vid_'+str(vid_num)+'.mp4').run()
    # shifting audio
    trim_audio('data/temp/'+tmp+'_aud_resampled.wav', 'data/temp/'+tmp+'.wav', 5.8, 10)
    vid = inp_unshift.video
    aud = ffmpeg.input('data/temp/'+tmp+'.wav')
    op_shift, err = ffmpeg.output(vid, aud.audio,
                                  path_shift+'vid_'+str(vid_num)+'.mp4').run()

    # clean up
    tmp = 'data/temp/'+tmp
    os.remove(tmp+'.mp4')
    os.remove(tmp+'_aud_resampled.wav')
    os.remove(tmp+'.wav')
    os.remove(tmp+'_aud.wav')


def download_video_parallel(yt_id, start_time, end_time, fps, aud_sampling, path_vid, path_shift,
                            path_unshift, rng):
    link = ["https://www.youtube.com/watch?v="+id for id in yt_id]
    Parallel(n_jobs=2, prefer="threads")(delayed(download_yt_link)(
        i+1, link[i], start_time[i], end_time[i], fps,
        aud_sampling, path_vid, path_shift, path_unshift, rng.rand()) for i in range(len(link)))
    # for i in range(len(link)):
    #     try:
    #         # downloading 360p video from YouTube
    #
    #         vid_num += 1
    #     except:
    #         print("Connection Error")  # to handle exception


def main():
    audioset_path = 'data/balanced_train_segments.csv'  # path to the list of YouTube videos
    audioset = pd.read_csv(audioset_path, quotechar='"',
                           skipinitialspace=True, skiprows=2)
    num_videos = 4
    rng = np.random.RandomState(seed=42)
    download_video_parallel(audioset.iloc[0:num_videos, 0], audioset.iloc[0:num_videos, 1],
                            audioset.iloc[0:num_videos, 2], 29.97, 22100, 'data/train/full_vid/',
                            'data/train/shifted/', 'data/train/unshifted/', rng)


if __name__ == '__main__':
    main()
