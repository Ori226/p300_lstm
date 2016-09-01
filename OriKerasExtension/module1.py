#%matplotlib inline
#import matplotlib.pyplot as plt
#plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\ori22_000\Downloads\ffmpeg-20150928-git-1d0487f-win64-static\ffmpeg-20150928-git-1d0487f-win64-static\bin\ffmpeg.exe"


#from tempfile import NamedTemporaryFile

#VIDEO_TAG = """<video controls>
# <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
# Your browser does not support the video tag.
#</video>"""

#def anim_to_html(anim):
#    if not hasattr(anim, '_encoded_video'):
#        with NamedTemporaryFile(suffix='.mp4') as f:
#            anim.save(f.name, fps=20, extra_args=['-vcodec', 'libx264'])
#            video = open(f.name, "rb").read()
#        anim._encoded_video = video.encode("base64")
    
#    return VIDEO_TAG.format(anim._encoded_video)

#from IPython.display import HTML

#def display_animation(anim):
#    plt.close(anim._fig)
#    return HTML(anim_to_html(anim))



#from matplotlib import animation

## First set up the figure, the axis, and the plot element we want to animate
#fig = plt.figure()
#ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
#line, = ax.plot([], [], lw=2)

## initialization function: plot the background of each frame
#def init():
#    line.set_data([], [])
#    return line,

## animation function.  This is called sequentially
#def animate(i):
#    x = np.linspace(0, 2, 1000)
#    y = np.sin(2 * np.pi * (x - 0.01 * i))
#    line.set_data(x, y)
#    return line,

## call the animator.  blit=True means only re-draw the parts that have changed.
#anim = animation.FuncAnimation(fig, animate, init_func=init,
#                               frames=100, interval=20, blit=True)

## call our new function to display the animation
#display_animation(anim)
#----------------


# %matplotlib qt 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

plt.rcParams['animation.ffmpeg_path'] = r"C:\Users\ori22_000\Downloads\ffmpeg-20150928-git-1d0487f-win64-static\ffmpeg-20150928-git-1d0487f-win64-static\bin\ffmpeg.exe"
plt.rcParams['animation.convert_path'] =r"C:\Program Files\ImageMagick-6.9.2-Q16\convert.exe"

def update_line(num, data, line):
     line.set_data(data[...,:num])
     return line,

fig1 = plt.figure()

data = np.random.rand(2, 25)
l, = plt.plot([], [], 'r-')
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel('x')
plt.title('test')
line_ani = animation.FuncAnimation(fig1, update_line, 25, fargs=(data, l),
    interval=50, blit=True)
#line_ani.save('lines.mp4')

from ThesisHelper import LoadSingleSubjectPython, readCompleteMatFile, ExtractDataVer2, ExtractDataVer3

def LoadSingleSubjectPythonNoPermute(file_name):
    
    res =  readCompleteMatFile(file_name);


    all_data, all_tags = ExtractDataVer3(res['all_relevant_channels'], res['marker_positions'], res['target'],0,400);

       
    trasposed_data = all_data.transpose(0,2,1)
    
    trasposed_data = trasposed_data.reshape(trasposed_data.shape[0],-1)


    all_target = trasposed_data[np.where(all_tags ==1 )[0],:]
    all_non_target = trasposed_data[np.where(all_tags != 1 )[0],:]

    subset_size = all_target.shape[0]

    all_target = all_target
    all_non_target = all_non_target
    return [all_target, all_non_target, res['marker_positions']]


[all_target, all_non_target, target_pos] = LoadSingleSubjectPythonNoPermute('C:\Users\ori22_000\Documents\Thesis\dataset\VPicr_11_03_03\RSVP_Color116msVPicr.mat')


fig2 = plt.figure()

x = np.arange(-9, 10)
y = np.arange(-9, 10).reshape(-1, 1)
base = np.hypot(x, y)
ims = []
for add in np.arange(15):
    
    ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),plt.title('blue should be up {0}'.format(add))))

FFwriter = animation.FFMpegWriter()

im_ani = animation.ArtistAnimation(fig2, ims, interval=50, repeat_delay=3000,
    blit=True)

#im_ani.save('im2.mp4', metadata={'artist':'Guido'}, writer = FFwriter)
im_ani.save('im2.gif', writer='imagemagick', fps=4)

plt.show()