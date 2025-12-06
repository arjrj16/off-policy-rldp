# 1. Set your data directory (adjust path as needed)
export DPPO_DATA_DIR=/home/melwani/67920/code2/off-policy-rldp/data  # or wherever your data is

cd $W
# 2. Create a temp directory for the download
mkdir -p tmp/robomimic_download

# 3. Download the transport HDF5 (proficient-human, low-dim)
cd tmp/robomimic_download
# wget https://diffusion-policy-data.s3.us-west-2.amazonaws.com/data/robomimic/datasets/transport/ph/low_dim.hdf5



# Transport PH low-dim (direct link from robomimic v0.1)
wget http://downloads.cs.stanford.edu/downloads/rt_benchmark/transport/ph/low_dim.hdf5

    
# 4. Go back to your project and run the processing script
cd $W/off-policy-rldp  # your project directory

# Then process
uv run python script/dataset/process_robomimic_dataset.py \
    --load_path $W/tmp/robomimic_download/low_dim.hdf5 \
    --save_dir $DPPO_DATA_DIR/robomimic/transport/ \
    --normalize

# 5. Clean up
# rm -rf /tmp/robomimic_download