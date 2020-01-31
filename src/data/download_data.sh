#!/bin/bash

DATADIR=data/raw/AerialImageDataset

which 7z >/dev/null || {
	echo 'you need 7z ; plz install it'
	echo 'ubuntu: sudo apt install p7zip-full'
	echo 'centos: sudo yum install p7zip p7zip-pluginsi -y'
	exit 1
}
which unzip >/dev/null || {
	echo 'you need unzip command ; plz install it'
	echo 'ubuntu: sudo apt install unzip'
	echo 'centos: sudo yum install zip unzip -y'
	exit 2
}

if [ ! -d "$DATADIR" ]
then

	cd data/raw/

	wget --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.001
	wget --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.002
	wget --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.003
	wget --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.004
	wget --no-check-certificate https://files.inria.fr/aerialimagelabeling/aerialimagelabeling.7z.005

	7z x aerialimagelabeling.7z.001
	unzip NEW2-AerialImageDataset.zip
	rm aerialimagelabeling.7z.*
	rm NEW2-AerialImageDataset.zip

	mv AerialImageDataset/train/* AerialImageDataset/
	mv AerialImageDataset/test/* AerialImageDataset/unlabeled/

	rm -r AerialImageDataset/test/
	rm -r AerialImageDataset/train/
  rm -r -i AerialImageDataset/unlabeled/

	cd ../../

else
  echo "$DATADIR already exists. Skipping downloading raw data."
fi
