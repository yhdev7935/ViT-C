
import random
from torchvision.transforms.functional import rotate 
class RandomRotate(object):
    """Rotate the image in a sample for data augmentation purposes. 

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        pass
        

    def __call__(self, image):
        # image, label = sample #sample['image'], sample['landmarks']
        
        rotations = [-270, -180, -90, 0, 90, 180, 270]

        return rotate(image, random.choice(rotations))#, label

class RandomCutout(object):
    """
    Performs a random cropping.
    """

    def __init__(self, cutout_scale = (0.0, 0.25)):
        self.cutout_scale = cutout_scale

    def __call__(self, image):
        h, w, c = image.shape 
        low_cutout, high_cutout = self.cutout_scale

        cutout_x = int(h*random.uniform(low_cutout, high_cutout))
        cutout_y = int(w*random.uniform(low_cutout, high_cutout))

        max_x = h - cutout_x
        max_y = w - cutout_y

        offset_x = int(random.uniform(0.0, max_x))
        offset_y = int(random.uniform(0.0, max_y))

        image[offset_x:offset_x + cutout_x, offset_y:offset_y + cutout_y, :] = 0 
        return image 