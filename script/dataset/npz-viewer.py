import numpy as np

# Use a context manager to ensure the file is closed properly
with np.load('/home/melwani/67920/code2/off-policy-rldp/data/robomimic/transport/normalization.npz') as data:
    # data is a dictionary-like object

    # Print a list of all array names (keys) in the file
    print("Arrays in the file:", data.files)

    # Iterate through the arrays and print their contents and shape
    for item in data.files:
        print(f"\n--- Array: {item} ---")
        print(data[item])  # Print the array content
        print(f"Shape of {item}: {data[item].shape}") # Print the array shape
        if(item == 'traj_lengths'):
            print(f"sum of traj_lengths: {np.sum(data[item])}")

    # You can also access a specific array directly by its key, e.g.
    # array_a = data['arr_0']
