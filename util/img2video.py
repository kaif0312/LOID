import cv2
import os

def create_video_from_images(image_folder, output_video_path, fps=30):
    # Get list of images in the folder
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    
    # Sort images by filename (assumes names like image0001.jpg, image0002.jpg, ...)
    images.sort()
    
    # Determine the width and height from the first image
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' or other codecs if necessary
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each image to the video
    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # Release the VideoWriter object
    video.release()
    print(f"Video saved as {output_video_path}")

# Example usage:
image_folder = '/home/kaifu10/Desktop/LaneLabor/Lane-Det-Abhiyaan/data'
output_video_path = 'insti.mp4'
fps = 15  # Frames per second

create_video_from_images(image_folder, output_video_path, fps)
