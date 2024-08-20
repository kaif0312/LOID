python test.py \
	--batchSize 1 \
	--nThreads 8 \
	--name objrmv \
	--dataset_mode testimage \
	--image_dir /home/kaifu10/Desktop/LANE_PAPER/CULANES/resized_occluded_frames \
	--mask_dir /home/kaifu10/Desktop/LANE_PAPER/CULANES/bounding_box_masks \
    --output_dir /home/kaifu10/Desktop/LANE_PAPER/crfill_new/CULANES_GT_pretrained_nodilation \
	--model inpaint \
	--netG baseconv \
	--which_epoch latest  \
	--load_baseg \
	$EXTRA
