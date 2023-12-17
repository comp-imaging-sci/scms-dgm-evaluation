from PIL import Image as pim
from time import time, strftime, localtime, perf_counter
from scipy import optimize, stats
from socket import gethostname
from matplotlib import pyplot
from skimage import io
import os
import numpy as np
import pandas as pd
import argparse

# Authors: Rucha Deshpande and Frank Brooks

def blockinate(arr, r, c):
	# Cut up the input array into rxc sized tiles
	R, C = arr.shape
	return arr.reshape(R//r, r, -1, c).swapaxes(1,2).reshape(-1, r, c)

def compute_weights_matrix(rows, cols, pbcs=False):
	# For fast computation of Moran's I, we'll need an RCxRC weights matrix to indicate pixel connectivity
	# this will be the same for all tiles and images
	# Could be made more efficient.
	ws = np.zeros((rows*cols, rows*cols))
	index_array = np.arange(rows*cols).reshape((rows, cols))

	for r in range(rows):
		for c in range(cols):
			if r != c:
				pixel = index_array[r, c]
				vnns = [(r, c-1), (r, c+1), (r-1, c), (r+1, c)]
				for nr, nc in vnns:
					if 0 <= nr < rows and 0 <= nc < cols:
						neighbor = index_array[nr, nc]
						ws[pixel, neighbor] = 1
	W = ws.sum()
	return ws/W


def get_masks(MASKS):
	# Load masks corresponding to the eight classes for later comparison
	masks = np.zeros((8, 256, 256), dtype= np.uint8)
	for idx in np.arange(0,8):
		masks[idx, :, :] = io.imread(os.path.join(MASKS, 'mask_%02d.png'%(idx))).astype(np.uint8)
	print(np.min(masks), np.max(masks))
	return masks

def get_true_class_vectors(masks, Z, BLOCK):
	# Get a binary vector representing class from each mask.
	# Cut up the mask into 16x16 tiles, compute intensity mean for each tile, decide if tile id foreground or background.
	# Create a vector for each mask. Each element of the binary vector represents a tile location.
	# That is, pattern match will be done on block-averaged, then thresholded vectors

	mask_vectors = np.zeros((Z, BLOCK**2), dtype=np.int16)
	for z in range(Z):
		tile_array = blockinate(masks[z], BLOCK, BLOCK)
		mean_array = np.mean(tile_array, axis=(1,2)).reshape((BLOCK,BLOCK))
		mean_array = np.round(mean_array)
		mask_vectors[z] = np.where(mean_array > 0, 1, 0).flatten()
	print(mask_vectors.shape)
	return mask_vectors

def calibrate_from_grayscale(Z, R, C):
	# The prescribed intensity distributions are employed to compute expected grayscale frequencies for foreground and background.
	# These distributions are also employed to determine the optimal threshold (FGTHRESH) that separates foreground from background.

	# These are the known distributions used to generate the data
	A, B, LOCATION, SCALE = 4, 2, 96, 152
	FOREGROUND_DIST = stats.beta(a=A, b=B, loc=LOCATION, scale=SCALE)

	A, B, LOCATION, SCALE = 2, 4, 8, 192
	BACKGROUND_DIST = stats.beta(a=A, b=B, loc=LOCATION, scale=SCALE)

	FG = R*C*80/256 # the number of foreground pixels in an image

	# Find what ideally ought to be more foreground than background
	GRAYS = np.arange(0,256)
	FREQS = np.round(FOREGROUND_DIST.pdf(GRAYS)*FG)

	lv = FREQS > 0
	FOREGROUND_GRAYS = GRAYS[lv]
	FOREGROUND_FREQUENCIES = np.round(FOREGROUND_DIST.pdf(FOREGROUND_GRAYS)*FG)

	BG = R*C - FG
	FREQS = np.round(BACKGROUND_DIST.pdf(GRAYS)*BG)
	lv = FREQS > 0
	BACKGROUND_GRAYS = GRAYS[lv]
	BACKGROUND_FREQUENCIES = np.round(BACKGROUND_DIST.pdf(BACKGROUND_GRAYS)*BG)

	def func(xs):
		return FOREGROUND_DIST.pdf(xs) - BACKGROUND_DIST.pdf(xs)

	sol = optimize.root(func, 150, method='lm')
	FGTHRESH = np.round(sol.x[0], 1) # This is the threshold between the foreground and background.
	# print(FGTHRESH) # 139.6
	return FOREGROUND_GRAYS, FOREGROUND_FREQUENCIES, BACKGROUND_GRAYS, BACKGROUND_FREQUENCIES, FGTHRESH

def compute_bad_tile_tolerance():
	# One method to compute how many bad tiles per image are acceptable before the whole image is declared "bad".
	# This method gives a tolerance of 3 tiles per image. This value could be manually changed by the user if a different tolerance is desired.
	dist = []
	for ii in np.arange(1000):
		tiles = []

		for i in np.arange(256):

			if random() < 1/256:

				tiles.append(1)
			else:
				tiles.append(0)
		dist.append(np.sum(tiles))

	return np.percentile(dist, 95) # this equals 3.

def check_forbidden_regions(MASKS_PATH, BLOCK, image_binary_vector):
	# Some tiles (at prescribed locations) are never foreground irrespective of class.
	# Check if this is true in the generated images.

	arr = np.invert(io.imread(os.path.join(MASKS_PATH, 'flags_forbidden_regions.png')))

	# Convert the forbidden region image to a binary vector indicating tile locations of forbidden regions.
	forbidden_mask_vectors = np.zeros((BLOCK**2), dtype=np.int16)
	forbidden_tile_array = blockinate(arr, BLOCK, BLOCK)
	forbidden_mean_array = np.round(np.mean(forbidden_tile_array, axis=(1,2)).reshape((BLOCK,BLOCK)))

	forbidden_mask_vectors = np.where(forbidden_mean_array > 0, 1, 0).flatten()
	idx = np.where(forbidden_mask_vectors == 0)

	# Check if the forbidden tile locations are ever violated.
	violated = np.sum(image_binary_vector[idx] + forbidden_mask_vectors[idx])

	return violated

def main():
	# This script analyzes all FLAG images in a folder (training data or generated data) and writes a dataframe.
	# A summary of the evaluation metrics computed from the dataframe will also be written out.
	# Images are expected to be .png. Extension may be changed by the user.
	# Each row in the dataframe corresponds to one image.
	# Columns in the summary dataframe are described below.

	# ===================================================================
	# col 1: "fn": image file name
	# col 2: pred.mask: predicted mask or class of the image (ranges form 0 to 7 - check masks to know which pattern is which class.)
	# cols 3 and 4: chi-squared statistic for the foreground (FG) and background (BG) prescribed intensity distributions
	# cols 5 and 6: no. of tiles per-image that violate the mean intensity range for FG and BG in the training data.
	# cols 7 and 8: no. of tiles per-image that violate Moran's I (pixel randomness) range for FG and BG in the training data
	# cols 9 to 16: error of the image mask w.r.t the mask from each of the seven classes. Zero error indicates the class with a perfect match.
	#               If no class has zero error, it indicates that the image has artifacts.
	#=====================================================================
	# This script also writes out lists of images with (1) perfect class identity, (2) ambiguous class identity, (3) violated forbidden regions.
	# A summary of the processed dataset is also written out at the end.
	#=====================================================================

	parser = argparse.ArgumentParser(description='Analysis of "flag" images')
	parser.add_argument('--id', type=str, default = 'reals', help='Training identifier. Use "reals" when using original data.')
	parser.add_argument('--images_path', type=str, required=True, help='Directory containing images to be analyzed.')
	parser.add_argument('--save', type=str, required=True, help='Directory where the generated dataframe should be saved.')
	parser.add_argument('--masks_path', type=str, required=True, help='Directory "flags_masks" containing the eight class masks.')
	args = parser.parse_args()

	KIND = args.id
	PATH = args.images_path
	SAVE = args.save
	MASKS_PATH = args.masks_path

	DUMP = os.path.join(SAVE, 'full_dataframes')
	DUMP2 = os.path.join(SAVE, 'check_image_lists')
	DUMP3 = os.path.join(SAVE, 'metrics')

	for ii in [SAVE, DUMP, DUMP2, DUMP3]:
		os.makedirs(ii, exist_ok=True)

	filenames = []
	for fn in os.listdir(PATH):
		if fn.endswith('.png'):
			filenames.append(fn)
	N = len(filenames)

	#=======================================================================================================================================
	# **If processing multiple datasets, this needs to be computed only once.** START.

	BLOCK = 16 # The flag is created on a grid of 16x16 pixels. Each tile is either background or foreground. This also aids automated analysis.
	bad_tile_tolerance = 3 # See function compute_bad_tile_tolerance. This value can be changed by the user if a different tolerance of bad tiles per image is desired.

	# These limits for the foreground and background intensity, and the tile-wise Moran's I are obtained from the training data.
	# Further description in the text of the main paper.
	FORE_LO, FORE_HI = 193, 202
	BACK_LO, BACK_HI = 66, 78
	MORAN_LO, MORAN_HI = -0.13322702161186606, 0.13322702161186606

	# Read class-specific masks. Transform them into binary vectors.
	masks = get_masks(MASKS_PATH)
	Z, R, C = masks.shape
	mask_vectors = get_true_class_vectors(masks, Z, BLOCK)
	# Process the prescribed intensity distributions to obtain expected frequencies for grayscale values in the background and foreground.
	FOREGROUND_GRAYS, FOREGROUND_FREQUENCIES, BACKGROUND_GRAYS, BACKGROUND_FREQUENCIES, FGTHRESH = calibrate_from_grayscale(Z, R, C)
	# Pre-compute the weight matrix for Moran's I.
	weights = compute_weights_matrix(BLOCK, BLOCK)

	COLUMNS = ['fn', 'pred.mask', 'chi.fore', 'chi.back', 'mean.out.fore', 'mean.out.back', 'moran.out.fore', 'moran.out.back']
	COLUMNS += ['error.%d' % e for e in range(Z)] # basically, the confusion vector
	TYPES = [np.uint32 if col in ['pred.mask', 'mean.out.fore', 'mean.out.back', 'moran.out.fore', 'moran.out.back'] else str if col in ['fn'] else np.float32 for col in COLUMNS]

	# **If processing multiple datasets, the section until here needs to be computed only once.** END.
	#===============================================================================================================================================

	data = pd.DataFrame(index=range(N), columns=COLUMNS)

	for col in ['mean.out.fore', 'mean.out.back', 'moran.out.fore', 'moran.out.back']:
		data[col].values[:] = 0

	BLOCK_TYPE = {1:'fore', 0:'back'}
	MEANBOUNDS = {'fore':(FORE_LO, FORE_HI), 'back':(BACK_LO, BACK_HI)}

	i = 0
	forbidden_region_violators = []
	for fn in filenames:

		data.loc[i]['fn'] = os.path.basename(fn)
		image_array = np.asarray(pim.open(os.path.join(PATH, fn)), dtype=np.uint8)

		# prepare generated image for block-by-block analysis
		tile_array = blockinate(image_array, BLOCK, BLOCK)
		mean_array = np.mean(tile_array, axis=(1,2)).reshape((BLOCK,BLOCK))

		# Convert the image to a binary vector and compare against all mask vectors to find the best match.
		binary_vector = np.where(mean_array > FGTHRESH, 1, 0).flatten()

		# Check if the image violates forbidden regions
		is_violated = check_forbidden_regions(MASKS_PATH, BLOCK, binary_vector)
		if is_violated:
			forbidden_region_violators.append(fn)

		errors = [] # Error w.r.t. each binary mask.
		for z in range(Z):
			err = np.abs(binary_vector - mask_vectors[z]).sum()/BLOCK**2
			errors.append((err, z))
			data.loc[i]['error.%d' % z] = err

		error, guess = sorted(errors)[0] # can be used to detect bad images -- find images that do not perfectly match any class.
		data.loc[i]['pred.mask'] = guess

		# now that we have a guess, use the mask to segment what *should* be foreground or background.
		# Impose foreground and background from the mask. This takes care of minor foreground mismatch and affords the DGM a fairer chance on the remaining tests.
		fg_lv = np.where(masks[guess] == 255)
		foreground_pixels = image_array[fg_lv]

		bg_lv = np.where(masks[guess] == 0)
		background_pixels = image_array[bg_lv]

		# compare the observed foreground to the expected foreground for that mask
		observed_frequencies = np.zeros(256)
		grays, counts = np.unique(foreground_pixels, return_counts=True)
		observed_frequencies[grays] = counts
		observed_frequencies = observed_frequencies[FOREGROUND_GRAYS] # number of pixels correctly declared foreground
		chisq, _ = stats.chisquare(observed_frequencies, FOREGROUND_FREQUENCIES)
		data.loc[i]['chi.fore'] = chisq

		# compare the observed background to the expected foreground for that mask
		observed_frequencies = np.zeros(256)
		grays, counts = np.unique(background_pixels, return_counts=True)
		observed_frequencies[grays] = counts
		observed_frequencies = observed_frequencies[BACKGROUND_GRAYS] # number of pixels correctly declared foreground
		chisq, _ = stats.chisquare(observed_frequencies, BACKGROUND_FREQUENCIES)
		data.loc[i]['chi.back'] = chisq

		# count the number of blocks with summary statistics that are outside the expected intervals
		means = np.mean(tile_array, axis=(1,2))
		sdevs = np.std(tile_array, axis=(1,2))

		for j in range(tile_array.shape[0]):
			# Process each image tile-wise. Recall tiles are 16x16 for all classes.
			# Count the no. of tiles per-image that violate Moran's co-efficient or the intensity mean.

			this_block_is = BLOCK_TYPE[mask_vectors[guess, j]]

			# compute moran coefficient and see if it is out of bounds
			z = (tile_array[j].flatten() - means[j])/sdevs[j]
			z_trans = z.reshape((-1, 1))
			moran = np.sum(z_trans*weights*z)

			if not ((MORAN_LO < moran) and (moran < MORAN_HI)):
				col = 'moran.out.%s' % this_block_is
				data.loc[i][col] += 1

			# see if mean is out of bounds for the expected type of block (FG or BG)
			lo, hi = MEANBOUNDS[this_block_is]
			if not ((lo < means[j]) and (means[j] < hi)):
				col = 'mean.out.%s' % this_block_is
				data.loc[i][col] += 1

		i += 1

	np.savetxt(os.path.join(DUMP2, '%s_forbidden-regions-violated.txt'%(KIND)), np.array(forbidden_region_violators), delimiter="\n", fmt="%s")
	print("List of images that violate forbidden regions is written out.")

	temp = dict(zip(COLUMNS, TYPES))
	data = data.astype(temp)

	data.to_csv(os.path.join(DUMP, 'flags_statistics_%s.csv' % KIND), index=False, float_format='%.6g')

	print("Dataframe has been written and saved. Printing summary...")

	#=============================================================================================================================================================================
	# Print summary of the dataframe
	print("====================================================================================================================")
	print("Dataset name: %s" %(KIND))

	df = pd.read_csv(os.path.join(DUMP, 'flags_statistics_%s.csv' % (KIND)), engine='python')

	# Check foreground pattern formation. Identify perfect/ambiguous images.
	df['min.error'] = df[['error.0','error.1', 'error.2', 'error.3', 'error.4', 'error.5', 'error.6', 'error.7']].min(axis=1)

	df_perfect = df[df['min.error']==0] # Filtered dataframe with only well-formed foreground patterns.

	df_perfect_class = df[df['min.error']==0]['fn'] # filenames of perfect rzns
	df_imperfect_class = df[df['min.error']!=0]['fn']

	num_perfect_rzns = len(df_perfect_class)
	frac_perfect_rzns = num_perfect_rzns / len(df)
	print("Percentage of images with perfect foreground patterns (class identity): %f (%%)"%(frac_perfect_rzns * 100))
	print("Writing list of images with: (1) perfect class identity (2) ambiguous class identity...")
	np.savetxt(os.path.join(DUMP2, '%s_correct-class-pattern.txt'%(KIND)), df_perfect_class, delimiter="\n", fmt="%s")
	np.savetxt(os.path.join(DUMP2, '%s_wrong-class-pattern.txt'%(KIND)), df_imperfect_class, delimiter="\n", fmt="%s")
	print('Done.')
	print("-------------------------------------------------------------------------------------------------------------------------------")

	# Print distribution of classes in the ensemble. All classes have equal prevalance in the training data.
	cls = df['pred.mask'].to_numpy()
	u, c = np.unique(cls, return_counts=True)
	print("Expected classes: 0-7 at equal prevalence. Distribution of classes in the processed dataset...")
	print("Class : count")
	for uu, cc in zip(u,c):
		print(uu,":", cc)
	print("-------------------------------------------------------------------------------------------------------------------------------")

	# Check intensity distribution of the foreground and background with the chi-squared statistic.
	chif = df_perfect['chi.fore'].to_numpy()
	chib = df_perfect['chi.back'].to_numpy()

	reject_chif = np.where(chif > 189.18699871308283) # see paper for method to determine threshold
	reject_chib = np.where(chib > 536.3072203449849)

	print("Percentage of images with incorrect FOREGROUND intensity distribution: %f (%%)"%( 100 * len(reject_chif[0])/num_perfect_rzns))
	print("Percentage of images with incorrect BACKGROUND intensity distribution: %f (%%)"%( 100 * len(reject_chib[0])/num_perfect_rzns))
	print("-------------------------------------------------------------------------------------------------------------------------------")

	# Check expected pixel randomness within each tile via Moran's I
	moranf = df_perfect['moran.out.fore'].to_numpy()
	moranb = df_perfect['moran.out.back'].to_numpy()

	reject_moranf = np.where(moranf > bad_tile_tolerance)
	reject_moranb = np.where(moranb > bad_tile_tolerance)

	print("Percentage of images with incorrect FOREGROUND pixel randomness: %f (%%)"%( 100 * len(reject_moranf[0])/num_perfect_rzns))
	print("Percentage of images with incorrect BACKGROUND pixel randomness: %f (%%)"%( 100 * len(reject_moranf[0])/num_perfect_rzns))
	print("-------------------------------------------------------------------------------------------------------------------------------")

	# Check expected mean intensity value within each tile in the FG or BG in an image.
	meanf = df_perfect['mean.out.fore'].to_numpy()
	meanb = df_perfect['mean.out.back'].to_numpy()
	reject_meanf = np.where(meanf > bad_tile_tolerance)
	reject_meanb = np.where(meanb > bad_tile_tolerance)

	print("Percentage of images with incorrect FOREGROUND mean intensity: %f (%%)"%( 100 * len(reject_meanf[0])/num_perfect_rzns))
	print("Percentage of images with incorrect BACKGROUND mean intensity: %f (%%)"%( 100 * len(reject_meanb[0])/num_perfect_rzns))
	print("=================================================================================================================================")

	#==============================================================================================================================================
	# Save all metrics.
	#============================================================================================================================================
	metrics = open(os.path.join(DUMP3,'flags_metrics_%s.txt'%(KIND)), 'w')
	metrics.write('# %s\n' % KIND )
	metrics.write('# percentage of images with ambiguous class identity : %.3f \n' %(100 * len(df_imperfect_class)/len(df)) )

	metrics.write('# percentage of images with incorrect foreground intensity distribution : %.3f \n' %(100 * len(reject_chif[0])/num_perfect_rzns))
	metrics.write('# percentage of images with incorrect background intensity distribution : %.3f \n' %(100 * len(reject_chib[0])/num_perfect_rzns))

	metrics.write('# percentage of images with incorrect foreground pixel randomness (Moran\'s I) : %.3f \n' %(100 * len(reject_moranf[0])/num_perfect_rzns))
	metrics.write('# percentage of images with incorrect background pixel randomness (Moran\'s I) : %.3f \n' %(100 * len(reject_moranb[0])/num_perfect_rzns))

	metrics.write('# distribution of classes (mean, stddev across all classes; expected values 0.125, 0) : %.3f, %3f \n' %(np.mean(c) / num_perfect_rzns, np.std(c) / num_perfect_rzns))
	metrics.close()

if __name__ == "__main__":
	main()
