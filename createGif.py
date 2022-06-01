import imageio
import os
from os import listdir
from os.path import isfile, join


filenames = [f for f in listdir('./GIFS/pngs') if isfile(join('./GIFS/pngs', f))]

# build gif
with imageio.get_writer('movie.gif', mode='I') as writer:
    for filename in filenames:
        path = f'./GIFS/pngs/{filename}'
        image = imageio.imread(path)
        writer.append_data(image)
        writer.append_data(image)
        writer.append_data(image)
        writer.append_data(image)
            
# Remove files
# for filename in set(filenames):
#     path = f'./GIFS/pngs/{filename}'
#     os.remove(path)