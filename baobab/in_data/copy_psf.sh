# Script to extract all the drizzled PSF maps from TDLMC rung 1 data
for d in tdlmc_data/rung2/code*/f160w-seed*; do cp -- "$d/drizzled_image/psf.fits" "psf_maps/psf_${d: -3}.fits"; done
