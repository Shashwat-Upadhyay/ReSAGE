import os
import json
import math
import numpy as np
import matplotlib.pyplot as plt

def inspect_directory(dir_path, name):
    print(f"Inspecting {name} directory: {dir_path}")
    if not os.path.exists(dir_path):
        print(f"Directory {dir_path} does not exist!")
        return None
    
    files = sorted([f for f in os.listdir(dir_path) if f.endswith('.npy')])
    num_files = len(files)
    
    if num_files == 0:
        print(f"No .npy files found in {dir_path}")
        return {
            "num_files": 0,
            "total_size_bytes": 0,
            "shapes": [],
            "dtypes": [],
            "min_val": None,
            "max_val": None,
            "mean_val": None,
            "std_val": None,
            "nan_count": 0,
            "inf_count": 0,
            "file_extensions": list(set([os.path.splitext(f)[1] for f in os.listdir(dir_path)])),
            "sample_filenames": []
        }
        
    total_size = sum(os.path.getsize(os.path.join(dir_path, f)) for f in files)
    
    # Initialize incremental stats
    first_file = os.path.join(dir_path, files[0])
    first_arr = np.load(first_file)
    shape_0 = first_arr.shape
    dtype_0 = str(first_arr.dtype)
    
    all_same_shape = True
    shapes_set = {shape_0}
    dtypes_set = {dtype_0}
    
    global_min = float('inf')
    global_max = float('-inf')
    
    total_pixels = 0
    sum_pixels = 0.0
    sum_sq_pixels = 0.0
    
    nan_count = 0
    inf_count = 0
    
    # We inspect all files incrementally to avoid RAM overhead
    for i, f in enumerate(files):
        arr = np.load(os.path.join(dir_path, f))
        
        shapes_set.add(arr.shape)
        dtypes_set.add(str(arr.dtype))
        
        # Check shapes and types
        if arr.shape != shape_0:
            all_same_shape = False
            
        # Clean array for calculations (min/max/mean/std should handle NaN/Inf)
        nans = np.isnan(arr)
        infs = np.isinf(arr)
        
        nan_count += np.sum(nans)
        inf_count += np.sum(infs)
        
        valid_mask = ~(nans | infs)
        valid_data = arr[valid_mask]
        
        if len(valid_data) > 0:
            file_min = np.min(valid_data)
            file_max = np.max(valid_data)
            if file_min < global_min:
                global_min = float(file_min)
            if file_max > global_max:
                global_max = float(file_max)
                
            total_pixels += len(valid_data)
            sum_pixels += float(np.sum(valid_data))
            sum_sq_pixels += float(np.sum(valid_data ** 2))
            
    # Final stats
    if total_pixels > 0:
        mean_val = sum_pixels / total_pixels
        var_val = (sum_sq_pixels / total_pixels) - (mean_val ** 2)
        std_val = math.sqrt(max(0.0, var_val))
    else:
        mean_val, std_val = 0.0, 0.0
        
    all_extensions = list(set([os.path.splitext(f)[1] for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]))
    
    info = {
        "num_files": num_files,
        "total_size_bytes": total_size,
        "shapes": [list(s) for s in shapes_set],
        "all_same_shape": all_same_shape,
        "dtypes": list(dtypes_set),
        "min_val": global_min if global_min != float('inf') else None,
        "max_val": global_max if global_max != float('-inf') else None,
        "mean_val": mean_val,
        "std_val": std_val,
        "nan_count": int(nan_count),
        "inf_count": int(inf_count),
        "file_extensions": all_extensions,
        "sample_filenames": files[:5]
    }
    
    return info

def main():
    base_dir = r"d:\My_Projects\ReSAGE"
    train_noisy_dir = os.path.join(base_dir, "data", "train", "NoisyLR")
    train_gt_dir = os.path.join(base_dir, "data", "train", "GT")
    test_noisy_dir = os.path.join(base_dir, "data", "test", "NoisyLR")
    
    # 1. Run inspections
    train_noisy_info = inspect_directory(train_noisy_dir, "Train NoisyLR")
    train_gt_info = inspect_directory(train_gt_dir, "Train GT")
    test_noisy_info = inspect_directory(test_noisy_dir, "Test NoisyLR")
    
    # 2. Check pairing
    train_noisy_files = sorted([f for f in os.listdir(train_noisy_dir) if f.endswith('.npy')])
    train_gt_files = sorted([f for f in os.listdir(train_gt_dir) if f.endswith('.npy')])
    
    is_perfectly_paired = (train_noisy_files == train_gt_files)
    
    # Scale factor
    scale_factor = None
    if train_noisy_info and train_gt_info:
        noisy_shape = train_noisy_info["shapes"][0]
        gt_shape = train_gt_info["shapes"][0]
        # Assuming height and width are last two dimensions
        scale_factor = gt_shape[-1] / noisy_shape[-1]
        
    # Compile report dictionary
    report = {
        "dataset_structure": {
            "train_noisy": train_noisy_info,
            "train_gt": train_gt_info,
            "test_noisy": test_noisy_info
        },
        "pairing_analysis": {
            "is_perfectly_paired": is_perfectly_paired,
            "num_noisy_files": len(train_noisy_files),
            "num_gt_files": len(train_gt_files),
            "sample_pairs_match": train_noisy_files[:10] if is_perfectly_paired else None
        },
        "scale_factor": scale_factor
    }
    
    # Write JSON report
    reports_dir = os.path.join(base_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    json_path = os.path.join(reports_dir, "dataset_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=4)
        
    # Write text report
    txt_path = os.path.join(reports_dir, "dataset_report.txt")
    with open(txt_path, "w") as f:
        f.write("==================================================\n")
        f.write("             ReSAGE Dataset Report                \n")
        f.write("==================================================\n\n")
        
        f.write("1. File Formats & Structures:\n")
        f.write(f"   - Stored format: Individual .npy (NumPy array) files\n")
        f.write(f"   - File extensions: {train_noisy_info['file_extensions']}\n\n")
        
        f.write("2. Sample counts:\n")
        f.write(f"   - Training samples: {train_noisy_info['num_files']}\n")
        f.write(f"   - Testing samples:  {test_noisy_info['num_files']}\n\n")
        
        f.write("3. Dimensions and Shapes:\n")
        f.write(f"   - NoisyLR shape: {train_noisy_info['shapes']}\n")
        f.write(f"   - GT shape:      {train_gt_info['shapes']}\n")
        f.write(f"   - All train samples same shape: NoisyLR: {train_noisy_info['all_same_shape']}, GT: {train_gt_info['all_same_shape']}\n")
        f.write(f"   - Scale factor (GT / NoisyLR): {scale_factor:.1f}x\n\n")
        
        f.write("4. Pairings:\n")
        f.write(f"   - Filename pairing matching: {is_perfectly_paired}\n")
        f.write(f"   - Sample noisy files: {train_noisy_info['sample_filenames']}\n")
        f.write(f"   - Sample GT files:    {train_gt_info['sample_filenames']}\n\n")
        
        f.write("5. Data Normalization & Ranges:\n")
        f.write("   Train NoisyLR:\n")
        f.write(f"     - Min: {train_noisy_info['min_val']}\n")
        f.write(f"     - Max: {train_noisy_info['max_val']}\n")
        f.write(f"     - Mean: {train_noisy_info['mean_val']:.6f}\n")
        f.write(f"     - Std:  {train_noisy_info['std_val']:.6f}\n")
        f.write(f"     - NaN count: {train_noisy_info['nan_count']}\n")
        f.write(f"     - Inf count: {train_noisy_info['inf_count']}\n")
        f.write("   Train GT:\n")
        f.write(f"     - Min: {train_gt_info['min_val']}\n")
        f.write(f"     - Max: {train_gt_info['max_val']}\n")
        f.write(f"     - Mean: {train_gt_info['mean_val']:.6f}\n")
        f.write(f"     - Std:  {train_gt_info['std_val']:.6f}\n")
        f.write(f"     - NaN count: {train_gt_info['nan_count']}\n")
        f.write(f"     - Inf count: {train_gt_info['inf_count']}\n")
        f.write("   Test NoisyLR:\n")
        f.write(f"     - Min: {test_noisy_info['min_val']}\n")
        f.write(f"     - Max: {test_noisy_info['max_val']}\n")
        f.write(f"     - Mean: {test_noisy_info['mean_val']:.6f}\n")
        f.write(f"     - Std:  {test_noisy_info['std_val']:.6f}\n")
        f.write(f"     - NaN count: {test_noisy_info['nan_count']}\n")
        f.write(f"     - Inf count: {test_noisy_info['inf_count']}\n\n")
        
        f.write("6. Summary of Findings:\n")
        f.write(f"   - Normalization: Intensity ranges are [Min: {train_noisy_info['min_val']}, Max: {train_noisy_info['max_val']}].\n")
        f.write(f"   - Missing/Malformed values: NaNs = {train_noisy_info['nan_count'] + train_gt_info['nan_count'] + test_noisy_info['nan_count']}; Infs = {train_noisy_info['inf_count'] + train_gt_info['inf_count'] + test_noisy_info['inf_count']}.\n")
        
    print(f"Reports saved to {reports_dir}")
    
    # 3. Create visualization reports/sample_pairs.png
    print("Generating sample pairs visualization...")
    num_visualized = 4
    fig, axes = plt.subplots(num_visualized, 2, figsize=(8, 4 * num_visualized))
    
    for i in range(num_visualized):
        filename = train_noisy_files[i]
        noisy_img = np.load(os.path.join(train_noisy_dir, filename))
        gt_img = np.load(os.path.join(train_gt_dir, filename))
        
        # If there are channels (e.g. C, H, W or H, W, C), handle them.
        # Let's check shape length:
        if len(noisy_img.shape) == 3:
            # Assume channel is first or last. If first, transpose to last for imshow.
            if noisy_img.shape[0] < noisy_img.shape[2]:
                noisy_disp = np.transpose(noisy_img, (1, 2, 0))
                gt_disp = np.transpose(gt_img, (1, 2, 0))
            else:
                noisy_disp = noisy_img
                gt_disp = gt_img
        else:
            noisy_disp = noisy_img
            gt_disp = gt_img
            
        axes[i, 0].imshow(noisy_disp, cmap='gray' if len(noisy_disp.shape) == 2 else None)
        axes[i, 0].set_title(f"NoisyLR: {filename} ({noisy_img.shape})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt_disp, cmap='gray' if len(gt_disp.shape) == 2 else None)
        axes[i, 1].set_title(f"GT: {filename} ({gt_img.shape})")
        axes[i, 1].axis('off')
        
    plt.tight_layout()
    png_path = os.path.join(reports_dir, "sample_pairs.png")
    plt.savefig(png_path)
    plt.close()
    print(f"Sample pairs plot saved to {png_path}")

if __name__ == "__main__":
    main()
