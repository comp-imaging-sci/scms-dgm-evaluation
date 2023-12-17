from PIL import Image as pim
from time import localtime, strftime, perf_counter
from scipy import stats
from socket import gethostname
from skimage import filters, measure
from matplotlib import pyplot

import os
import numpy as np
from scipy.stats import chisquare
import argparse
import pandas as pd

# Authors: Rucha Deshpande and Frank Brooks

def calibrate_error_scale_from_perfect_templates(TEMPLATE_PATH):
	TEMPLATES = {}

	for fn in os.listdir(TEMPLATE_PATH):
		if fn.endswith('.png'):
			image_array = np.asarray(pim.open(os.path.join(TEMPLATE_PATH, fn))).astype(np.int32)
			name = fn[:-4]
			TEMPLATES[name] = {'imarr': image_array}

	# Calibrate the original templates. Compare all combinations of letters to obtain a confusion matrix
	L = len(TEMPLATES)
	diffs = np.zeros((L, L))

	for i in range(L):
		t1 = sorted(TEMPLATES)[i]
		for j in range(L):
			t2 = sorted(TEMPLATES)[j]
			diffs[i,j] = np.sum(np.abs(TEMPLATES[t1]['imarr'] - TEMPLATES[t2]['imarr']))

	# Get letter pairs that are most different. Establish an error scale from 0 to MAXDIFF.
	a = np.argmax(diffs)
	r, c = np.unravel_index(a, diffs.shape)
	MAXDIFF = diffs[r,c]

	return MAXDIFF, TEMPLATES

def prepare_to_scan(R, C, S, FREQS):
	# expected location of the window centers

	rows = np.arange(S, R, 2*S)
	cols = np.arange(S, C, 2*S)
	centers = np.dstack(np.meshgrid(cols, rows)).reshape(-1, 2)

	COLUMNS = ['fn']
	COLUMNS += sorted(FREQS)
	COLUMNS += ['Z.V', 'Z.W', 'Z.K', 'X.Y', 'num.unc', 'tot.unc', 'max.unc']

	return centers, COLUMNS

def process_image_array(PATH, fn, TEMPLATES, MAXDIFF, centers, S, dump, kind, FREQS, COLUMNS):

	# Read image
	image_array = np.asarray(pim.open(os.path.join(PATH, fn)))

	# Set count to zero for all letters.
	observed_frequencies = dict([(s, 0) for s in FREQS])
	predictions = {}
	symbol_vector, uncertainty_vector = [], []

	# Bins for the uncertainty vector
	bins = np.append(np.linspace(0,0.5,16), np.inf)

	# Template matching at each letter location in the image. Yields best guess for letter identity and its uncertainty.
	for (col, row) in centers:
		tile = image_array[row-S:row+S,col-S:col+S]

		scores = []
		for name in TEMPLATES:
			diff = np.sum(np.abs(TEMPLATES[name]['imarr'] - tile))
			scores.append((diff, name))

		scores.sort()
		guess = scores[0][1]
		symbol_vector.append(guess)
		observed_frequencies[guess] += 1
		uncertainty_vector.append(np.sum(np.abs(TEMPLATES[guess]['imarr'] - tile))/MAXDIFF)

	if int(np.sum(uncertainty_vector)) == 0: # explicit condition so that uncertainty_vector doesn't become a vector of -1: [-1, -1, ..].
		uncertainty_vector = ''.join(['0']*len(uncertainty_vector))
	else:
		uncertainty_vector = np.digitize(uncertainty_vector, bins, right=True) - 1
		uncertainty_vector = ''.join(list(map(lambda x:x[2:], map(hex, uncertainty_vector))))

	# Compute chi-square goodness-of-fit statistic from the expected and observed frequencies of letters.
	f_exp = np.array(list(FREQS.values())) # expected frequencies
	f_obs = np.array(list(observed_frequencies.values())) # observed frequencies
	chisq, _ = chisquare(f_obs, f_exp, ddof=7)

	predictions[fn] = {'word': ''.join(symbol_vector), 'uncertainty': uncertainty_vector, 'chisq': chisq}

	# Check horizontal and vertical letter-pairs. Get their count.
	observed_frequencies.update({'Z.V':0, 'Z.W':0, 'Z.K':0, 'X.Y':0})
	hparts = [predictions[fn]['word'][i:i+8] for i in range(0, len(predictions[fn]['word']), 8)]
	vparts = [predictions[fn]['word'][i::8] for i in range(0, 8)]

	xy = []
	[xy.append(i.count("XY")) for i in hparts if 'XY' in i]; observed_frequencies['X.Y'] = np.sum(xy)

	zv, zw, zk = [],[],[]
	[zv.append(i.count("ZV")) for i in vparts if 'ZV' in i]; observed_frequencies['Z.V'] = np.sum(zv)
	[zw.append(i.count("ZW")) for i in vparts if 'ZW' in i]; observed_frequencies['Z.W'] = np.sum(zw)
	[zk.append(i.count("ZK")) for i in vparts if 'ZK' in i]; observed_frequencies['Z.K'] = np.sum(zk)

	# Update all predictions for the chosen image.
	predictions[fn].update(observed_frequencies)

	# Summary stats for uncertainty. Number of non-zero values: num.unc, total uncertainty over all letters: tot.unc, maximum uncertainty for a letter: max.unc
	# Convert uncertainty_vector from hexadecimal to decimal array. Check for non-zero values. Summarize.
	temp = np.array([int('0x%s' % u, 16) for u in predictions[fn]['uncertainty']])
	lv = temp > 0
	if temp[lv].size !=0 :
		predictions[fn].update({'num.unc':lv.sum(), 'tot.unc':temp[lv].sum(), 'max.unc':temp[lv].max()})
	else:
		predictions[fn].update({'num.unc':lv.sum(), 'tot.unc':lv.sum(), 'max.unc':lv.sum()})

	# Write out image filename and all other columns for this image.
	dump.write('%s' % fn)
	for c in COLUMNS[1:]:
		dump.write(',%s' % predictions[fn][c])
	dump.write(',%s,%s,%f\n' % (predictions[fn]['word'], predictions[fn]['uncertainty'], predictions[fn]['chisq']))
	return dump

def main():

	# This script analyzes all ALPHABET images in a folder (training data or generated data) and writes a dataframe.
	# A summary of the evaluation metrics computed from the dataframe will also be written out.
	# Images are expected to be .png. Extension may be changed by the user but may require calibration and visual check for template matching.
	# Each row in the dataframe corresponds to one image.
	# Columns in the summary dataframe are described below.
	# ===================================================================
	# col 1: "fn": image file name
	# cols 2 to 9: per-image count of individual letters: H,K,L,V,W,X,Y,Z,
	# cols 10 to 13: per-image count of letter pairs: 'Z.V', 'Z.W', 'Z.K', 'X.Y'
	# col 14: num.unc: count of letters with non-zero uncertainty. If zero, this indicates all letters are perfectly formed in an image.
	# col 15: tot.unc: summation of uncertainty of all letters. Can be used as a threshold to exclude images from further analysis.
	# col 16: max.unc: uncertainty of the worst formed letter in the image. Can be used as a threshold to exclude images from further analysis.
	# col 17: word: all letters in the image in row-wise order. Can be used to test memorization via string comparison with the training data.
	# col 18: uncertainty: uncertainty vector in hexadecimal scale. Each value matches a letter in the "word".
	# Generally, values above b in hexadecimal (i.e., 12 in decimal) are not visually recognizable.
	# col 19: chisq: Chi-square goodness-of-fit statistic. Checks letter prevalence, not letter-pair prevalence.
	# Determines acceptability of a realization based on letter prevalence - intuitively, ensemble correctness, not per-image correctness.
	# Value 0 indicates perfect prevalence, value above 14.067 indicates unacceptable letter prevalence.
	#=====================================================================
	#=====================================================================

	parser = argparse.ArgumentParser(description='Analysis of alphabet images')
	parser.add_argument('--id', type=str, default = 'reals', help='Training identifier. Use "reals" when using original data.')
	parser.add_argument('--images_path', type=str, required=True, help='Directory containing images to be analyzed.')
	parser.add_argument('--template_path', type=str, required=True, help='Directory "letter_templates" containing the unique letter templates.')
	parser.add_argument('--save', type=str, required=True, help='Directory where the generated dataframe should be saved. ')
	args = parser.parse_args()

	CODE = 'alphabet.py'

	HOST = str(gethostname()).split('.')[0]

	kind = args.id # training identifier
	PATH = args.images_path
	TEMPLATE_PATH = args.template_path
	SAVE = args.save

	DUMP = os.path.join(SAVE, 'full_dataframes')
	DUMP2 = os.path.join(SAVE, 'check_image_lists')
	DUMP3 = os.path.join(SAVE, 'metrics')

	for ii in [SAVE, DUMP, DUMP2, DUMP3]:
		os.makedirs(ii, exist_ok=True)

	FREQS = {'V': 1, 'W': 1, 'K': 2, 'Z': 4, 'X': 8, 'Y': 8, 'L': 16, 'H': 24} # encoded letter frequencies in each realization
	R, C = 256, 256 # image size
	S = 16 # letter template size//2

	MAXDIFF, TEMPLATES = calibrate_error_scale_from_perfect_templates(TEMPLATE_PATH)
	print("Calibration complete.") # This needs to be done only once even if analyzing multiple datasets in a loop.
	centers, COLUMNS = prepare_to_scan(R, C, S, FREQS)

	dump = open(os.path.join(DUMP,'alphabet_statistics_%s.txt'%(kind)), 'w')
	temp = ','.join([c for c in COLUMNS + ['word', 'uncertainty', 'chisq']])
	dump.write('%s\n' % temp)

	START = perf_counter()

	for fn in os.listdir(PATH):
		if fn.endswith('.png'):
			dump = process_image_array(PATH, fn, TEMPLATES, MAXDIFF, centers, S, dump, kind, FREQS, COLUMNS)

	CPU = perf_counter() - START
	NOW = strftime('%A %Y %B %d %H:%M:%S %Z', localtime())
	FOOTER = '\n# %s\n# %s --- cpu: %.1f (s)\n# %s\n' % (NOW, CODE, CPU, HOST)
	dump.write('%s' % FOOTER)
	dump.close()
	print("Dataframe has been written and saved. Printing summary...")

	#==========================================================================================================================
	# Summarize dataframe. Prints summary and saves lists of perfect/imperfect images that could be visually spot checked.
	#==========================================================================================================================

	df0 = pd.read_csv(os.path.join(DUMP, 'alphabet_statistics_%s.txt'%(kind)), skipfooter=3, engine='python')

	# Exclude realizations with badly formed letters if required.
	# E.g., use condition: max.unc <= c to filter out realizations with at least on letter that is beyond visual recognition.
	df = df0[df0['max.unc']<=12] # Retain only realizations that are visually perfect for further analyses.

	chizero_rzns = np.count_nonzero(df['chisq'].to_numpy() == 0) # count "trues" or nonzeros for the condition arr==0.
	chi95_rzns = len(np.where(df['chisq'].to_numpy() <= 14.067)[0]) # critical value of chi-sq dist at prob 0.95 for dof=7.

	print("====================================================================================================================")
	print("Dataset name: %s" %(kind))
	print("Visually acceptable fraction of realizations: %f" %(len(df[df['max.unc']<=12])/ len(df))) # Lower than 12 indicates visually recognizable letters. Threshold is visually subjective.
	print("Ensemble prevalence test: Contextually acceptable fraction of realizations: %f" %(chi95_rzns/ len(df))) # Via chi square goodness-of-fit, at 95% critical value.
	print("Per-realization prevalence test: Contextually **perfect** fraction of realizations: %f" %(chizero_rzns/ len(df)))
	print("====================================================================================================================")


	xy_mean, xy_stddev = df['X.Y'].mean(), df['X.Y'].std()
	zv_mean, zv_stddev = df['Z.V'].mean(), df['Z.V'].std()
	zw_mean, zw_stddev = df['Z.W'].mean(), df['Z.W'].std()
	zk_mean, zk_stddev = df['Z.K'].mean(), df['Z.K'].std()

	print("Paired prevalences in training data: XY-8, ZV-1, ZW-1, ZK-2 ")
	print("Paired prevalences in processed data (mean, std.dev.). XY: %.2f, %.2f, ZV: %.2f, %.2f, ZW: %.2f, %.2f, ZK: %.2f, %.2f" %(xy_mean, xy_stddev, zv_mean, zv_stddev, zw_mean, zw_stddev, zk_mean, zk_stddev))
	print("====================================================================================================================")

	chizero_fns = df.loc[df['chisq'] == 0]['fn']
	chizero_reject_fns = df.loc[df['chisq'] != 0]['fn']
	chi95_reject_fns = df.loc[df['chisq'] > 14.067]['fn']
	visually_bad_fns = df.loc[df['max.unc'] >= 12]['fn']

	print("Writing out filenames of contextually **incorrect** images.")
	np.savetxt(os.path.join(DUMP2, '%s_incorrect-per-image-context.txt'%(kind)), chizero_reject_fns, delimiter="\n", fmt="%s")
	np.savetxt(os.path.join(DUMP2, '%s_chisquared-rejected-images.txt'%(kind)), chi95_reject_fns, delimiter="\n", fmt="%s")

	print("Done. Writing out filenames of contextually **perfect** images.")
	np.savetxt(os.path.join(DUMP2, '%s_perfect-context.txt'%(kind)), chizero_fns, delimiter="\n", fmt="%s")

	print("Done. Writing out filenames of images with at least one visually unrecognizable letter.")
	np.savetxt(os.path.join(DUMP2, '%s_unrecognizable-letter.txt'%(kind)), visually_bad_fns, delimiter="\n", fmt="%s")

	print("Done!")

	#==============================================================================================================================================
	# Save all metrics.
	#============================================================================================================================================
	metrics = open(os.path.join(DUMP3,'alphabet_metrics_%s.txt'%(kind)), 'w')
	metrics.write('# %s\n' % kind )
	metrics.write('# percentage of images where ALL letters in an image are not visually recognizable : %.3f \n' %(100 * len(df[df['max.unc']>12])/len(df)) )
	metrics.write('# percentage of images that are rejected based on ensemble prevalence : %.3f \n' %(100 * len(chi95_reject_fns)/len(df)))
	metrics.write('# percentage of images that are rejected based on per-image prevalence of letters : %.3f \n' %(100 * len(chizero_reject_fns)/len(df)))
	metrics.close()

if __name__ == "__main__":
	main()
