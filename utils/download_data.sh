# Download everything
wget --show-progress -O data/hotspots-data.tar.gz https://www.cs.utexas.edu/~tushar/interaction-hotspots/hotspot_data.tar.gz
echo "Data downloaded. Extracting files..."

tar -zxvf data/hotspots-data.tar.gz
rm -r data/hotspots-data.tar.gz