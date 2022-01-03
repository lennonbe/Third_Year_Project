import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import cv2
import matplotlib.image as mpimg
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage import data, img_as_float
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed, chan_vese, active_contour, flood, flood_fill
from skimage.filters import gaussian
import skimage.filters as filters
from skimage.segmentation import mark_boundaries
from PIL import Image

def chan_vese_seg(image):
    # Feel free to play around with the parameters to see how they impact the result
    cv = chan_vese(image, mu=0.25, lambda1=1, lambda2=1, tol=1e-3, max_iter=200,
                   dt=0.5, init_level_set="checkerboard", extended_output=True)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.flatten()

    ax[0].imshow(image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("Original Image", fontsize=12)

    ax[1].imshow(cv[0], cmap="gray")
    ax[1].set_axis_off()
    title = "Chan-Vese segmentation - {} iterations".format(len(cv[2]))
    ax[1].set_title(title, fontsize=12)

    ax[2].imshow(cv[1], cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title("Final Level Set", fontsize=12)

    ax[3].plot(cv[2])
    ax[3].set_title("Evolution of energy over iterations", fontsize=12)


    fig.tight_layout()
    plt.show()

def all_others(image, image2):

    img = img_as_float(image)
    img2 = img_as_float(image2)

    segments_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    print(type(segments_fz))
    print(segments_fz)
    segments_slic = slic(img, n_segments=250, compactness=10, sigma=1,
                         start_label=1)
    segments_quick = quickshift(img, kernel_size=3, max_dist=6, ratio=0.5)
    gradient = sobel(rgb2gray(img))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)

    print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')
    print(f'SLIC number of segments: {len(np.unique(segments_slic))}')
    print(f'Quickshift number of segments: {len(np.unique(segments_quick))}')

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(mark_boundaries(img, segments_fz))
    ax[0, 0].set_title("Felzenszwalbs's method")
    ax[0, 1].imshow(mark_boundaries(img, segments_slic))
    ax[0, 1].set_title('SLIC')
    ax[1, 0].imshow(mark_boundaries(img, segments_quick))
    ax[1, 0].set_title('Quickshift')
    ax[1, 1].imshow(mark_boundaries(img, segments_watershed))
    ax[1, 1].set_title('Compact watershed')

    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax2[0, 0].imshow(mark_boundaries(img2, segments_fz))
    ax2[0, 0].set_title("Felzenszwalbs's method GT")
    ax2[0, 1].imshow(mark_boundaries(img2, segments_slic))
    ax2[0, 1].set_title('SLIC GT')
    ax2[1, 0].imshow(mark_boundaries(img2, segments_quick))
    ax2[1, 0].set_title('Quickshift GT')
    ax2[1, 1].imshow(mark_boundaries(img2, segments_watershed))
    ax2[1, 1].set_title('Compact watershed GT')

    for a in ax.ravel():
        a.set_axis_off()

    for b in ax2.ravel():
        b.set_axis_off()

    plt.tight_layout()
    plt.show()

#all_others(myImage, myImage2)

def get_data():

    gt = []
    rgb = []
    for e in os.listdir('./Temp_Images/RGB'):
        rgb.append(Image.open('./Temp_Images/RGB/' + e).resize((256, 256), Image.ANTIALIAS))

    for e in os.listdir('./Temp_Images/GT'):
        gt.append(Image.open('./Temp_Images/GT/' + e).resize((256, 256), Image.ANTIALIAS))

    return gt, rgb

input1, input2 = get_data()

def felzenszwalb_seg(images, images2):

    float_images = []
    segments_fz = []
    for e in images:

        temp = img_as_float(e)

        float_images.append(temp)

        segments_fz.append(felzenszwalb(temp, scale=500000, sigma=0.5, min_size=0))

    float_images_gt = []
    for e in images2:
        temp = img_as_float(e)

        float_images_gt.append(temp)


    print(f'Felzenszwalb number of segments: {len(np.unique(segments_fz))}')

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(mark_boundaries(float_images[0], segments_fz[0]))
    ax[0, 0].set_title("IMG 1 - Felzenszwalb RGB")
    ax[0, 1].imshow(mark_boundaries(float_images[1], segments_fz[1]))
    ax[0, 1].set_title('IMG 2 - Felzenszwalb RGB')
    ax[1, 0].imshow(mark_boundaries(float_images[2], segments_fz[2]))
    ax[1, 0].set_title('IMG 3 - Felzenszwalb RGB')
    ax[1, 1].imshow(mark_boundaries(float_images[3], segments_fz[3]))
    ax[1, 1].set_title('IMG 4 - Felzenszwalb RGB')

    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax2[0, 0].imshow(mark_boundaries(float_images_gt[0], segments_fz[0]))
    ax2[0, 0].set_title("IMG 1 - Felzenszwalb GT")
    ax2[0, 1].imshow(mark_boundaries(float_images_gt[1], segments_fz[1]))
    ax2[0, 1].set_title('IMG 2 - Felzenszwalb GT')
    ax2[1, 0].imshow(mark_boundaries(float_images_gt[2], segments_fz[2]))
    ax2[1, 0].set_title('IMG 3 - Felzenszwalb GT')
    ax2[1, 1].imshow(mark_boundaries(float_images_gt[3], segments_fz[3]))
    ax2[1, 1].set_title('IMG 4 - Felzenszwalb GT')

    for a in ax.ravel():
        a.set_axis_off()

    for b in ax2.ravel():
        b.set_axis_off()

    plt.tight_layout()
    plt.show()

def slic_seg(images, images2):

    float_images = []
    segments_fz = []
    for e in images:

        temp = img_as_float(e)

        float_images.append(temp)

        segments_fz.append(slic(temp, n_segments=250, compactness=10, sigma=1,
                         start_label=1))

    float_images_gt = []
    for e in images2:
        temp = img_as_float(e)

        float_images_gt.append(temp)


    print(f'SLIC number of segments: {len(np.unique(segments_fz))}')

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(mark_boundaries(float_images[0], segments_fz[0]))
    ax[0, 0].set_title("IMG 1 - SLIC RGB")
    ax[0, 1].imshow(mark_boundaries(float_images[1], segments_fz[1]))
    ax[0, 1].set_title('IMG 2 - SLIC RGB')
    ax[1, 0].imshow(mark_boundaries(float_images[2], segments_fz[2]))
    ax[1, 0].set_title('IMG 3 - SLIC RGB')
    ax[1, 1].imshow(mark_boundaries(float_images[3], segments_fz[3]))
    ax[1, 1].set_title('IMG 4 - SLIC RGB')

    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax2[0, 0].imshow(mark_boundaries(float_images_gt[0], segments_fz[0]))
    ax2[0, 0].set_title("IMG 1 - SLIC GT")
    ax2[0, 1].imshow(mark_boundaries(float_images_gt[1], segments_fz[1]))
    ax2[0, 1].set_title('IMG 2 - SLIC GT')
    ax2[1, 0].imshow(mark_boundaries(float_images_gt[2], segments_fz[2]))
    ax2[1, 0].set_title('IMG 3 - SLIC GT')
    ax2[1, 1].imshow(mark_boundaries(float_images_gt[3], segments_fz[3]))
    ax2[1, 1].set_title('IMG 4 - SLIC GT')

    for a in ax.ravel():
        a.set_axis_off()

    for b in ax2.ravel():
        b.set_axis_off()

    plt.tight_layout()
    plt.show()

def quickshift_seg(images, images2):

    float_images = []
    segments_fz = []
    for e in images:

        temp = img_as_float(e)

        float_images.append(temp)

        segments_fz.append(quickshift(temp, kernel_size=3, max_dist=6, ratio=0.5))

    float_images_gt = []
    for e in images2:
        temp = img_as_float(e)

        float_images_gt.append(temp)


    print(f'Quickshift number of segments: {len(np.unique(segments_fz))}')

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(mark_boundaries(float_images[0], segments_fz[0]))
    ax[0, 0].set_title("IMG 1 - Quickshift RGB")
    ax[0, 1].imshow(mark_boundaries(float_images[1], segments_fz[1]))
    ax[0, 1].set_title('IMG 2 - Quickshift RGB')
    ax[1, 0].imshow(mark_boundaries(float_images[2], segments_fz[2]))
    ax[1, 0].set_title('IMG 3 - Quickshift RGB')
    ax[1, 1].imshow(mark_boundaries(float_images[3], segments_fz[3]))
    ax[1, 1].set_title('IMG 4 - Quickshift RGB')

    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax2[0, 0].imshow(mark_boundaries(float_images_gt[0], segments_fz[0]))
    ax2[0, 0].set_title("IMG 1 - Quickshift GT")
    ax2[0, 1].imshow(mark_boundaries(float_images_gt[1], segments_fz[1]))
    ax2[0, 1].set_title('IMG 2 - Quickshift GT')
    ax2[1, 0].imshow(mark_boundaries(float_images_gt[2], segments_fz[2]))
    ax2[1, 0].set_title('IMG 3 - Quickshift GT')
    ax2[1, 1].imshow(mark_boundaries(float_images_gt[3], segments_fz[3]))
    ax2[1, 1].set_title('IMG 4 - Quickshift GT')

    for a in ax.ravel():
        a.set_axis_off()

    for b in ax2.ravel():
        b.set_axis_off()

    plt.tight_layout()
    plt.show()

def watershed_seg(images, images2):

    float_images = []
    segments_fz = []
    for e in images:

        temp = img_as_float(e)

        float_images.append(temp)

        gradient = sobel(rgb2gray(temp))
        segments_fz.append(watershed(gradient, markers=250, compactness=0.001))

    float_images_gt = []
    for e in images2:
        temp = img_as_float(e)

        float_images_gt.append(temp)


    print(f'Watershed number of segments: {len(np.unique(segments_fz))}')

    fig, ax = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax[0, 0].imshow(mark_boundaries(float_images[0], segments_fz[0]))
    ax[0, 0].set_title("IMG 1 - Compact Watershed RGB")
    ax[0, 1].imshow(mark_boundaries(float_images[1], segments_fz[1]))
    ax[0, 1].set_title('IMG 2 - Compact Watershed RGB')
    ax[1, 0].imshow(mark_boundaries(float_images[2], segments_fz[2]))
    ax[1, 0].set_title('IMG 3 - Compact Watershed RGB')
    ax[1, 1].imshow(mark_boundaries(float_images[3], segments_fz[3]))
    ax[1, 1].set_title('IMG 4 - Compact Watershed RGB')

    fig2, ax2 = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)

    ax2[0, 0].imshow(mark_boundaries(float_images_gt[0], segments_fz[0]))
    ax2[0, 0].set_title("IMG 1 - Compact Watershed GT")
    ax2[0, 1].imshow(mark_boundaries(float_images_gt[1], segments_fz[1]))
    ax2[0, 1].set_title('IMG 2 - Compact Watershed GT')
    ax2[1, 0].imshow(mark_boundaries(float_images_gt[2], segments_fz[2]))
    ax2[1, 0].set_title('IMG 3 - Compact Watershed GT')
    ax2[1, 1].imshow(mark_boundaries(float_images_gt[3], segments_fz[3]))
    ax2[1, 1].set_title('IMG 4 - Compact Watershed GT')

    for a in ax.ravel():
        a.set_axis_off()

    for b in ax2.ravel():
        b.set_axis_off()

    plt.tight_layout()
    plt.show()

#quickshift_seg(input1, input2)

img = Image.open('./Temp_Images/RGB/01042016_Choctawhatchee_River_near_Bellwood_AL.tif_RGB.pngpatch_38_22.png').resize((256, 256), Image.ANTIALIAS)
#img = rgb2gray(img)

# Generate noisy synthetic data
data = img_as_float(img)
data_sobel = filters.sobel(data[..., 0])

# Run random walker algorithm
x = 100
y = 200
seed_point = (x, y)  # Experiment with the seed point
flood_mask = flood(data_sobel, seed_point, tolerance=1)  # Experiment with tolerance

'''
fig, ax = plt.subplots(nrows=3, figsize=(10, 20))

ax[0].imshow(data)
ax[0].set_title('Original')
ax[0].axis('off')

ax[1].imshow(data_sobel)
ax[1].set_title('Sobel filtered')
ax[1].axis('off')

ax[2].imshow(data)
ax[2].imshow(data_sobel, cmap=plt.cm.gray, alpha=0.3)
ax[2].plot(y, x, 'wo')  # seed point
ax[2].set_title('Segmented with `flood`')
ax[2].axis('off')

fig.tight_layout()
plt.show()
'''

'''
#OpenCV find contours segmentation
path = './Temp_Images/RGB/01042016_Choctawhatchee_River_near_Bellwood_AL.tif_RGB.pngpatch_38_22.png'
img = cv2.imread(path)
img = cv2.resize(img,(256,256))

gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
_,thresh = cv2.threshold(gray, np.mean(gray), 255, cv2.THRESH_BINARY_INV)
edges = cv2.dilate(cv2.Canny(thresh,0,255),None)

cnt = sorted(cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2], key=cv2.contourArea)[-1]
mask = np.zeros((256,256), np.uint8)
masked = cv2.drawContours(mask, [cnt],-1, 255, -1)

dst = cv2.bitwise_and(img, img, mask=mask)
segmented = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

plt.imshow(segmented,aspect="auto")
plt.show()
'''

#K-Means Clustering for segmentation (region based)
pic = plt.imread('./Temp_Images/RGB/TX_20160811_WaterExtent_WetSoil_Khartoum.tif_RGB.png')  # png values already normalized to 0-1 but if jpg need to divide by 255
print(pic.shape)
plt.imshow(pic)

pic_n = pic.reshape(pic.shape[0]*pic.shape[1], pic.shape[2])

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
kmeans = KMeans(n_clusters=15, random_state=0).fit(pic_n)
#sc = SpectralClustering(3, affinity='precomputed', n_init=100,assign_labels='discretize').fit(pic_n)
#ac = AgglomerativeClustering().fit(pic_n)

pic2show = kmeans.cluster_centers_[kmeans.labels_]

cluster_pic = pic2show.reshape(pic.shape[0], pic.shape[1], pic.shape[2])
plt.imshow(cluster_pic)
plt.show()
