import imageio
import sys

def convert(filename):
    # Convert the generated video to an animated GIF using imageio
    video_path = filename + '.mp4'
    output_gif_path = filename + '.gif'
    video_reader = imageio.get_reader(video_path)
    fps = video_reader.get_meta_data()['fps']

    # Write the frames to a GIF file
    imageio.mimsave(output_gif_path, [frame for frame in video_reader], fps=fps, compress='lossless')

def main():
    num_args = len(sys.argv)
    # Get the arguments from the command line
    if (num_args != 2):
        print ("python ImageIOTool.py filename (no extension)")
        return
    filename = sys.argv[1]
    print ("Convert video: " + filename)
    convert(filename)
    return

if __name__ == "__main__":
    # If the script is run directly, call the main function
    main()
