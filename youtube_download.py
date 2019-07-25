from pytube import YouTube
from tqdm import tqdm

#file_size=0
save_path ="/home/ahsangoheer/Documents/Tools/Videos"
# def progress_Check(stream = None, chunk = None, file_handle = None, remaining = None):
# 	#Gets the percentage of the file that has been downloaded.
# 	    percent = (100*(remaining-file_size))/file_size
# 	    print("{:00.0f}% downloaded".format(percent))
    
mylinks=list(open('links.txt','r')) 



for link in mylinks:
    try:
        yt = YouTube(link) #,on_progress_callback=progress_Check)
        print("Downloading: {} \n".format(yt.title))
    except:
        print('There was an error while connecting!')

    streams = yt.streams.all()
    for i in streams:
        print(str(i)+'\n')
    itag = input('Enter the itag number of the desired format : ')
    desired_stream=yt.streams.get_by_itag(itag)
    desired_stream.download(save_path)
    print('{} Downloaded!'.format(yt.title))
    file_size=0

print('Process Complete! Files are located in {}'.format(save_path))


