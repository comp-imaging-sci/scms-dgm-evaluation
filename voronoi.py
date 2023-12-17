import skan
import numpy as np
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from skimage import morphology, io
from skan.image_stats import image_summary

from skimage.filters import threshold_sauvola
from skimage import data, measure
from skimage.measure import label, regionprops

import scipy.stats
from time import localtime, strftime, perf_counter
from socket import gethostname

import argparse

# Authors: Rucha Deshpande and Frank Brooks

CODE = 'voronoi.py'
HOST = str(gethostname()).split('.')[0]

properties = ['area', 'mean_intensity']

def intensity_var(region, intensities):
	# Grayscale variance in a single Voronoi region. Ideally, should be zero.
	return np.var(intensities[region])

def get_skeleton(shading_type, imin):
	# Post-processing to extract skeleton.
	# Sauvola thresholding seems to work well with generated images (ProGAN, SG2, DDPM) as compared to other thresholding methods from skimage.
	# But visually test this if your network produces unusual images.
	if shading_type == 's': # Using shaded Voronoi - the default dataset
		imin = np.where(imin < threshold_sauvola(imin, 5), 1, 0) # Empirically set to 5.
	else: # Using unshaded Voronoi, only used for a demo.
		imin = np.where(imin < 64, 1, 0) # top 2% as threshold
	return morphology.skeletonize(imin)

def get_bw_stats(skel_in, df_summary):
	# Skeleton statistics from a thresholded image: no. of junctions, jxns/area, mean branch length between junctions, and "squiggle".
	# Refer https://skeleton-analysis.org/stable/ or https://github.com/jni/skan for details on skeleton statistics.

	stats0 = image_summary(skel_in) # default stats from skan

	skel = skan.Skeleton(skel_in) # skeleton object
	stats1 = skan.summarize(skel) # summarize skeleton object, gives additional stats as compared to stats0

	df_summary['num_junctions'] = stats0['number of junctions']
	df_summary['junction_density'] = stats0['junctions per unit area']
	squiggle = np.log2( stats1['branch-distance'] / stats1['euclidean-distance'] )

	df_summary['mean-squiggle'] = squiggle.mean()
	df_summary['stddev-squiggle'] = squiggle.std()

	j2j = stats1[stats1['branch-type'] == 2] # distance between two junctions in a skeleton
	df_summary['mean-J2J-branch-distance'] = (j2j['branch-distance'].mean())
	df_summary['stddev-J2J-branch-distance'] = (j2j['branch-distance'].std())

	return df_summary

def get_region_stats(imin, skeletonized, shading_type, kind, fn, properties, df_summary):

	# Extract Voronoi regions and get a summary of the region properties per-image.
	# Get masked regions
	dilated_skel = morphology.dilation(skeletonized, morphology.disk(2))
	bw_regions = np.invert(dilated_skel)

	labeled_image = measure.label(bw_regions)
	df_summary['num_regions'] = np.max(labeled_image)

	# Extract properties
	table = measure.regionprops_table(labeled_image, imin, properties=properties, extra_properties=[intensity_var])
	df_summary['area_mean'] = table['area'].mean(); df_summary['area_std'] = table['area'].std()
	# Get the grayscale variance in each Voronoi region in the image (ideally, zero). Can be used to identify images with artifacts.
	# Compute the mean and variance within one image. Helps identify images with major artifacts.
	df_summary['mean_region_gs_variance'] = table['intensity_var'].mean()
	df_summary['std_region_gs_variance'] = table['intensity_var'].std()

	if shading_type == 's':
		# Compute correlation of grayscale with area. Should ideally be 1.
		df_summary['rho'], _ = scipy.stats.spearmanr(table['mean_intensity'], table['area'])
		df_summary['tau'], _ = scipy.stats.kendalltau(table['mean_intensity'], table['area'])
	else:
		# Doesn't make sense to compute for this for the unshaded Voronoi dataset.
		df_summary['tau'] = 0; df_summary['rho'] = 0

	if kind == 'reals':
		df_summary['class_idx'] = int(fn.split('-')[0])
	else:
		# Number of voronoi regions per-image determines class.
		df_summary['class_idx'] = df_summary['num_regions']

	return df_summary

def main():
	# =========================================================================================================================================================================
	# This script analyzes all VORONOI images in a folder (training data or generated data) and writes a dataframe.
	# A summary of the evaluation metrics computed from the dataframe will also be written out.
	# Images are expected to be .png. Extension may be changed by the user but may require calibration and visual check for the post-hoc segmenter.
	# Each row in the dataframe corresponds to one image.
	# Columns in the summary dataframe are described below.

	# ==========================================================================================================================================================================
	# col 1: "fn": image file name
	# cols 2 to 7: per-image skeleton statistics: number of junctions in the skeleton, density of junctions, mean and std deviation of "squiggliness" or non-straight edges,
	#              mean and std deviation of branch length between two junctions.
	#              For more details, refer https://skeleton-analysis.org/stable/ or https://github.com/jni/skan
	# cols 8 to 10: per-image region statistics: number of regions, mean and std deviation of all region areas per-image.
	#              Other statistics could also be computed - such as region convex hull, eccentricity etc.
	# cols 11 - 12: per-image mean and std deviation of the per-region grayscale variance.
	#             This helps identify images with shading artifacts - in both the shaded and unshaded version of the Voronoi SCM.
	# cols 13-14: For use only with the shaded Voronoi. Represents the rank correlation between grayscale and area. Can be used to identify imperfect images using a threshold.
	#             True rho is 1 for the training data.
	# col 15: class_idx. This is the same as the number of regions. Can be used to calibrate a post-hoc region segmenter on the training data.
	#             For the training data, the class_idx value is taken from the file name. Col 8: num_regions is computed from the post-hoc segmenter.
	#             The difference between the two gives a measure of error for the post-hoc segmenter.

	#===========================================================================================================================================================================
	# This script also writes out lists of images with (1) potential shading artifacts, (2) incorrect shading rules or correlation
	# A plot of the ensemble class distribution is also written out. Helps identify class interpolation or extrapolation or mode collapse in the generated ensemble
	#=============================================================================================================================================================================

	parser = argparse.ArgumentParser(description='Analysis of Voronoi images')
	parser.add_argument('--id', type=str, default = 'reals', help='Training identifier. Use "reals" when using original data.')
	parser.add_argument('--images_path', type=str, required=True, help='Directory containing images to be analyzed.')
	parser.add_argument('--save', type=str, required=True, help='Directory where the generated dataframe should be saved. ')
	parser.add_argument('--corr_threshold', type=float, default=0.8, help='[0,1]. Threshold (lower: more tolerant) that determines acceptability in terms of rank correlation. Training dataset has a rank correlation of 1.')
	parser.add_argument('--artifact_threshold', type=float, default=10,
						help='Threshold (greater: more tolerant of artifacts) that determines acceptability in terms of grayscale variance within a Voronoi region. Training dataset has a value of 0.')
	args = parser.parse_args()

	kind = args.id
	path = args.images_path
	save = args.save
	shading_type = 's' # indicates shaded Voronoi, as opposed to unshaded Voronoi used for a demonstration in the main paper.

	DUMP = os.path.join(save, 'full_dataframes')
	DUMP2 = os.path.join(save, 'check_images_list')
	DUMP3 = os.path.join(save, 'plots')
	DUMP4 = os.path.join(save, 'metrics')

	for ii in [save, DUMP, DUMP2, DUMP3, DUMP4]:
		os.makedirs(ii, exist_ok = True)

	COLUMNS = ['fn','num_junctions', 'junction_density',	'mean-squiggle', 'stddev-squiggle',
	'mean-J2J-branch-distance',	'stddev-J2J-branch-distance', 'num_regions',
	'area_mean', 'area_std', 'mean_region_gs_variance', 'std_region_gs_variance', 'tau', 'rho', 'class_idx']

	dump = open(os.path.join(DUMP,'%svoronoi_statistics_%s.txt'%(shading_type, kind)), 'w')
	temp = ','.join([c for c in COLUMNS])
	dump.write('%s\n' % temp)

	START = perf_counter()
	for fn in os.listdir(path)[:30]:

		if fn.endswith('.png'):

			predictions = {}
			df_summary = {}

			predictions[fn] = df_summary

			im = io.imread(os.path.join(path, fn))
			skeletonized = get_skeleton(shading_type, im)

			df_summary = get_bw_stats(skeletonized, df_summary)
			df_summary = get_region_stats(im, skeletonized, shading_type, kind, fn, properties, df_summary)

			dump.write('%s' % fn)
			for c in COLUMNS[1:]:
				dump.write(',%f' % predictions[fn][c])
			dump.write('\n')

	CPU = perf_counter() - START
	NOW = strftime('%A %Y %B %d %H:%M:%S %Z', localtime()) # more informative formatting
	FOOTER = '\n# %s\n# %s --- cpu: %.1f (s)\n# %s\n' % (NOW, CODE, CPU, HOST)
	dump.write('%s' % FOOTER)
	dump.close()

	print("Dataframe written and saved. Printing summary...\n")
	# ===============================================================================================================================================
	df = pd.read_csv(os.path.join(DUMP, '%svoronoi_statistics_%s.txt'%(shading_type, kind)), skipfooter=3, engine='python')
	regions = df['num_regions'].to_numpy()

	print("Expected classes: 16, 32, 48, 64 at equal prevalence. Plotting distribution of classes in the processed dataset...\n")
	plt.figure()
	sns.kdeplot(regions)
	for xx in [16, 32, 48, 64]:
		plt.plot(xx, 0.001, "rD") # Markers for true class location
	plt.xticks([0, 16, 32, 48, 64, 80, 96])
	plt.xlabel('Class/ number of regions per-image')
	plt.title('Class distribution in ensemble: %s \n Expected: Four classes - 16,32,48,64 - at equal prevalence \n'%(kind))
	plt.savefig(os.path.join(DUMP3, 'class_distribution_%s.png'%(kind)))
	plt.close()
	print("Done.\n")

	reject_images_corr = len(df.loc[df['rho'] < args.corr_threshold]) / len(df)
	reject_fns_corr = df.loc[df['rho'] < args.corr_threshold]['fn']
	np.savetxt(os.path.join(DUMP2, '%s_low-grayscale-area-correlation.txt'%(kind)), reject_fns_corr, delimiter="\n", fmt="%s")
	print("Percentage of realizations that are **below** the correlation acceptability threshold of %f: %f %% \n" %(args.corr_threshold, reject_images_corr * 100))
	print("List of realizations that have low value of prescribed correlation written out. These might be images with low or weird contrast. Confirm with visual checks.\n")

	print("Writing out a list of realizations that probably have artifacts. Confirm with visual checks.\n")
	artifact_fns = df.loc[df['std_region_gs_variance'] > args.artifact_threshold]['fn']
	np.savetxt(os.path.join(DUMP2, '%s_probable-artifacts.txt'%(kind)), artifact_fns, delimiter="\n", fmt="%s")

	print("Done!")

	#==============================================================================================================================================
	# Save all metrics.
	#============================================================================================================================================
	metrics = open(os.path.join(DUMP4,'%svoronoi_metrics_%s.txt'%(shading_type, kind)), 'w')

	metrics.write('# %s\n' % kind )
	metrics.write('# percentage of images that are below the acceptable correlation value %f : %.3f \n' %(args.corr_threshold, reject_images_corr * 100) )

	metrics.write('# percentage of images with high grayscale variance (above %f) per-image : %.3f \n' %(args.artifact_threshold, 100 * len(artifact_fns)/len(df) ))
	metrics.write('# distribution of classes (mean, stddev over number of regions per-image) : %.3f, %3f \n' % (np.mean(regions), np.std(regions)))
	metrics.close()

if __name__ == "__main__":
	main()
