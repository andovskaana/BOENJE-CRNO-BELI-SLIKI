import numpy as np
import cv2

def load_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Konvertiranje na slika vo float32 i golemina 224x224
    img = img.astype(np.float32) / 255.0
    img = cv2.resize(img, (224, 224))

    return img


# Pateki
prototxt_path = 'colorization_deploy_v2.prototxt'
caffemodel_path = 'colorization_release_v2.caffemodel'
pts_in_hull_path = 'pts_in_hull.npy'

# Load na veke istreniran model
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
pts = np.load(pts_in_hull_path)

# Dodadi cluster centari kako 1x1 convolutions na modelot
class8 = net.getLayerId('class8_ab')
conv8 = net.getLayerId('conv8_313_rh')
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype(np.float32)]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]

image_path = 'img_1.png'
image = load_image(image_path)

# Odzemi mean  vrednost za boenje
image_lab = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
image_l = image_lab[:, :, 0]  # Extract the L channel

# Pripremi L kanal za input
image_l = image_l - 50
net.setInput(cv2.dnn.blobFromImage(image_l))

# Predviduvanje na ab kanali
ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))

# Resize na ab kanali da se so golemina kako originalnata slika
(ab_height, ab_width) = ab_channel.shape[:2]
ab_channel = cv2.resize(ab_channel, (image.shape[1], image.shape[0]))

# Konkateniraj L i ab kanali
lab_image = np.concatenate((image_lab[:, :, 0][:, :, np.newaxis], ab_channel), axis=2)

# Vrati vo RGB
color_image = cv2.cvtColor(lab_image, cv2.COLOR_Lab2RGB)
color_image = np.clip(color_image, 0, 1)
color_image = (255 * color_image).astype(np.uint8)

# Save i prikazi
cv2.imwrite('colorized_image2.jpg', color_image)
cv2.imshow('Colorized Image', color_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
