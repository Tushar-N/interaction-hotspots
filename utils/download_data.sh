# Download everything
wget --show-progress -O data/hotspots-data.tar.gz https://utexas.box.com/shared/static/j4ets5j30c9gacjkmpve29jd3hsdkcmu.gz
echo "Data downloaded. Extracting files..."

tar -zxvf data/hotspots-data.tar.gz
rm -r data/hotspots-data.tar.gz
