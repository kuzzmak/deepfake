from pytube import YouTube


if __name__ == '__main__':

    link = 'https://www.youtube.com/watch?v=5InzccRssnE'
    filename = 'trump.mp4'
    save_path = r'C:\Users\kuzmi\Documents\deepfake\data'

    yt = YouTube(link)
    stream = yt.streams \
        .filter(progressive=True, file_extension='mp4') \
        .order_by('resolution') \
        .desc() \
        .first()
    stream.download(save_path, filename)
