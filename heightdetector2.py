# Program for person and face detection 
import boto3
from pprint import pprint
import cv2
import time
from moviepy.editor import VideoFileClip

def boundingBox(img):
    """
    avg: calcluates the average of each column
    """
    avg = [sum(col)/len(col) for col in zip(*box)]
    #for x,y,w,h in box:
    cv2.rectangle(img, (round(avg[0]),round(avg[1])), (round(avg[0]+avg[2]),\
                        round(avg[1]+avg[3])), (0,0,255), 1)
    return img

def boundingBox2(img):
    avg2 = [sum(col)/len(col) for col in zip(*box2)]
    #for x,y,w,h in box2:
    cv2.rectangle(img, (round(avg2[0]),round(avg2[1])), (round(avg2[0]+avg2[2]),\
                        round(avg2[1]+avg2[3])), (255,0,0), 1)
    return img  
    
def writeVideo(path1, path2, BBox): 
    """
    Extract a frame and draw the average bounding box over each frame
    """
    clip1=VideoFileClip(path1)
    video_clip=clip1.fl_image(BBox)
    video_clip.write_videofile(path2, audio=False)

client = boto3.client('rekognition',region_name='us-east-1')
                         
t0 = time.time()
response = client.start_person_tracking(
    Video={
        'S3Object': {
            'Bucket': 'height-detector-test',
            'Name': 'man_standing.mp4'
        }
    }
)
t1 = time.time()
print(t1-t0)
response4 = client.get_person_tracking(JobId=response['JobId'],SortBy='TIMESTAMP') 
# make sure to include SNS AMazon to get notification that the process is successful
pprint(response4)
#Timestamp in milliseconds

while(response4['JobStatus'] == 'IN_PROGRESS'):
    response4 = client.get_person_tracking(JobId=response['JobId'],SortBy='TIMESTAMP')

t2 = time.time()
print('success:',t2-t1)

# Bounding box
box = []
frameHeight = response4['VideoMetadata']['FrameHeight']
frameWidth = response4['VideoMetadata']['FrameWidth']
for data in response4['Persons']:
    y = round(data['Person']['BoundingBox']['Top']*frameHeight)
    x = round(data['Person']['BoundingBox']['Left']*frameWidth)
    h = round(data['Person']['BoundingBox']['Height']*frameHeight)
    w = round(data['Person']['BoundingBox']['Width']*frameWidth)
    box.append([x,y,w,h])

box2 = []
for data in response4['Persons']:
    if 'Face' in data['Person']:
        y2 = round(data['Person']['Face']['BoundingBox']['Top']*frameHeight)
        x2 = round(data['Person']['Face']['BoundingBox']['Left']*frameWidth)
        h2 = round(data['Person']['Face']['BoundingBox']['Height']*frameHeight)
        w2 = round(data['Person']['Face']['BoundingBox']['Width']*frameWidth)
        box2.append([x2,y2,w2,h2])
    
outputVideo = 'D:/curvai/rekognition/data/man_standing_o.mp4'
inputVideo = 'D:/curvai/rekognition/data/man_standing.mp4'
writeVideo(inputVideo,outputVideo, boundingBox)

outputVideo2 = 'D:/curvai/rekognition/data/man_standing_o2.mp4'
inputVideo2 = 'D:/curvai/rekognition/data/man_standing_o.mp4'
writeVideo(inputVideo2,outputVideo2,boundingBox2)

