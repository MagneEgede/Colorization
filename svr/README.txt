in Path locations in trainset.npy and metrics_compare.npy you have to modify the paths to match where images are
then run trainset.npy where you can modify these two values:
	specify how many % of data with subtrainsize
	specify now many number of images will be tested on with subtestset
then run train.npy
	it should save a model file, add that path to the metrics_compare.npy
	you can change number of segments, size of segment and if it uses %subtrainsize or 80% training data
run metrics_compare.npy
	if plotting is true, it wil generate various plots of image_nr
	if plotting is false it will generate error metrics on the the first n=testset images that appear in trainset