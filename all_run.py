from utils import *

if __name__ == "__main__": 

    args["data_dir"] = r"/mnt/research/research/Data/MyocardialScarDefns_kycn/Data"         # Main data directory
    args["output_dir"] = r"/mnt/research/research/Data/MyocardialScarDefns_kycn/outs"       # output files save directory
    args["patient_folder_name"] = ["p_140"]                                                 # if you want to work in some spesific files, fill the names
    args["patient_folder_overwrite"] = 1                                                    # 1 means -> if the same output folder name in the output path, pass that file
    args["total_seg_model"] = "heartchambers_highres"                                       # Total segmentator model name
    args["segmentation_mask_onoff"] = 1                                                     # segmentation_mask -> open/close choosing myocardium segmentation filter for multiclass segmentation files
    args["segmentation_mask_min"] = 0                                                       # for multiclass segmentation files
    args["segmentation_mask_max"] = 2                                                       # for multiclass segmentation files
    args["erosion_onoff"] = 1                                                               # use/ do not use erosion
    args["erosion_disk_r_mm"] = 0.5                                                         # erosion disk radius
    args["erosion_run_n_times"] = 2                                                         # use erosion that times
    args["clustering_type"] = 3                                                             # 1 - cylindirical in 2 axis, 2 - kmeans, 3 - use equipotential surfaces in SAX thickness and use cylindirical in 4CH, else do not use clustering
    args["segment_seed_number"] = 54                                                        # total number of segments used for clustering
    args["segment_nz_number"] = 18                                                          # n_z -> number of slices in z direction for clustering = 1, total number of slices in 4CH cylindirical direction for clustering = 3
    args["k_means_3d"] = 1                                                                  # k_means_3d -> 1 - use 3D kmeans algorithm, 0 - use 2D
    args["laplace_max_iter"] = 5000                                                         # laplace solver max iteration
    args["laplace_tolerance"] = 1e-5                                                        # laplace solver tolerance

    print('CUDA available:', torch.cuda.is_available())
    torch.version.cuda
    print(torch.__version__)

    base_root_dir = args["data_dir"]          

    if args["patient_folder_name"] == []:
        all_patients = os.listdir(base_root_dir)            
    else:                                         
        all_patients = args["patient_folder_name"]   

    print(f"All patient list: {all_patients}")
    output_save_path = args["output_dir"]         

    if_folder_exist_overwrite = args["patient_folder_overwrite"]   

    for p in all_patients:

        full_path = os.path.join(output_save_path, p)
        root_dir = os.path.join(base_root_dir, p)

        print("Output save directory:", repr(full_path))  # debug

        if os.path.exists(full_path):
            if if_folder_exist_overwrite:
                outputs_file_path = os.path.join(output_save_path, p) + "/"
            else:
                print(f"Path and files exist, it will not work again! {os.path.join(output_save_path, p)}")
                continue
        else:
            os.makedirs(full_path, exist_ok=True) # create output file
            outputs_file_path = full_path + "/"

        print(f"Output file path: {outputs_file_path}")
        print(f"Processing patient: {root_dir}")
    
        dicom_files_name_list, data_file_path = list_dicom_files(root_dir)  # find all dicom files in directory
        print(f"Data file path: {data_file_path}")

        # Check dcm/nifti values if intensity values the same as HU
        check_dcm = dicom_files_name_list[1]
        print(f"Checking DICOM file: {check_dcm}")
        dcm = pyd.dcmread(check_dcm)

        slope = float(dcm.RescaleSlope)
        intercept = float(dcm.RescaleIntercept)

        print(f"Slope: {slope} / Intercept: {intercept}")

        raw = dcm.pixel_array.astype(np.int32)

        slope = float(dcm.RescaleSlope)
        intercept = float(dcm.RescaleIntercept)

        hu = raw * slope + intercept

        y, x = raw.shape[0]//2, raw.shape[1]//2
        print(f"raw: {raw[y, x]} / hu_from_dcm: {hu[y, x]}")

        img = sitk.ReadImage(check_dcm)
        arr = sitk.GetArrayFromImage(img)       # (1, y, x) or (y, x)

        sitk_val = arr[0, y, x] if arr.ndim==3 else arr[y, x]

        print(f"sitk value: {sitk_val}") 

        # Check if intensities are also HU values
        if sitk_val != hu[y, x]: 
            print(f"Becareful!!!!!!!!! Intensities {sitk_val} are not equal to HU {hu[y, x]} values!") 

        # Read .dcm files and combine to create nifti
        dicom_files = data_file_path
        print(f"Data directory: {dicom_files}")
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_files)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        print(f"Image size: {image.GetSize()}")
        print(f"Image spacing: {image.GetSpacing()}")
        print(f"Image origin: {image.GetOrigin()}")
        print(f"Image direction: {image.GetDirection()}")

        # For transform dcm to nifti, we can use sitk.GetArrayFromImage to get the 3D array and then save it as nifti using sitk.WriteImage. However, if we want to crop the image in z dimension, we can directly use sitk.RegionOfInterest to crop the image and then save it as nifti.
        #
        # 

        array = sitk.GetArrayFromImage(image)

        size = image.GetSize()  # (x,y,z)

        # (x_start, y_start, z_start)
        start = [0, 0, 0]

        # (x_size, y_size, z_size)
        # if you want to crop slices in axises, use numbers instead of sizes
        new_size = [size[0], size[1], size[2]]

        cropped = sitk.RegionOfInterest(image, new_size, start)

        sitk.WriteImage(cropped, outputs_file_path + "/converted_image.nii.gz")
        # Save nifti

        input_file = outputs_file_path + "/converted_image.nii.gz"
        output_file = outputs_file_path + "/segmentations"

        # Using TotalSegmentator...
        out_segmentation = totalsegmentator(
            input=input_file,
            output=output_file,
            fast=False, 
            preview = False,
            task=args["total_seg_model"], 
            ml=True, 
            verbose=True
        )

        print(f"Solutions in there: {output_file}")

        print(f"Segmentations mapping: {class_map[args["total_seg_model"]]}")

        # post_segmentation_processing is for erosion, segmentation selection and clustering
        out_arr, erosion_out, spacing, origin, direction, image, roi = post_segmentation_processing(input_file, 
                                                                                                    output_file, 
                                                                                                    segmentation_mask = args["segmentation_mask_onoff"], 
                                                                                                    segmentation_mask_min = args["segmentation_mask_min"],
                                                                                                    segmentation_mask_max = args["segmentation_mask_max"],
                                                                                                    erosion_on = args["erosion_onoff"], 
                                                                                                    r_mm = args["erosion_disk_r_mm"], 
                                                                                                    repeat_alg = args["erosion_run_n_times"], 
                                                                                                    clustering = args["clustering_type"], 
                                                                                                    seed_k = args["segment_seed_number"], 
                                                                                                    k_means_3d = args["k_means_3d"], 
                                                                                                    n_z = args["segment_nz_number"],
                                                                                                    laplace_max_iter  = args["laplace_max_iter"],
                                                                                                    laplace_tolerance = args["laplace_tolerance"])

        # Calculate and print voxel sizes for each of the segmentation label
        voxel_labels = out_arr
        print(f"Unique Voxel Labels: {np.unique(voxel_labels)}")
        seed_index = np.unique(voxel_labels)
        mask_list = [np.sum(voxel_labels == i) for i in seed_index]
        print(f"Unique Voxel Label Sizes: {mask_list}")
        
        #plt.figure(figsize=(10,5))
        #plt.bar(seed_index[1:], mask_list[1:])  # Skip the 0 label (background)
        #plt.xlabel("Cluster Label")
        #plt.ylabel("Voxel Count")
        #plt.title("Voxel Count per Cluster")
        #plt.show()
        #mask_list

        # --- write output ---
        out_sitk = sitk.GetImageFromArray(out_arr)   # array (z,y,x)
        out_sitk.SetSpacing(spacing)
        out_sitk.SetOrigin(origin)
        out_sitk.SetDirection(direction)

        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation('RAS')
        out_sitk = orienter.Execute(out_sitk)

        sitk.WriteImage(out_sitk, outputs_file_path + "/postproc_2d.nii.gz")
        print("saved:", outputs_file_path + "/postproc_2d.nii.gz")
        ####
        out_sitk = sitk.GetImageFromArray(erosion_out.astype(np.uint16)) 
        out_sitk.SetSpacing(spacing)
        out_sitk.SetOrigin(origin)
        out_sitk.SetDirection(direction)

        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation('RAS')
        out_sitk = orienter.Execute(out_sitk)
        sitk.WriteImage(out_sitk, outputs_file_path + "/erosion_out.nii.gz")
        print("saved:", outputs_file_path + "/erosion_out.nii.gz")
        ####
        out_sitk = sitk.GetImageFromArray(roi.astype(np.uint16)) 
        out_sitk.SetSpacing(spacing)
        out_sitk.SetOrigin(origin)
        out_sitk.SetDirection(direction)

        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation('RAS')
        out_sitk = orienter.Execute(out_sitk)
        sitk.WriteImage(out_sitk, outputs_file_path + "/roi.nii.gz")
        print("saved:", outputs_file_path + "/roi.nii.gz")
        ####

        # Print voxel counts, percentiles
        print(f'image.shape: {image.shape}')
        #plt.plot(intens)
        #plt.show

        print(f'One point HU check Array {array[array.shape[0]//2, array.shape[1]//2, 0]}')
        print(f'One point HU check CT {image[image.shape[0]//2, image.shape[1]//2,0]}')

        masked_intensities_erosion_out = image[erosion_out]
        voxel_cnt = np.prod(image.shape)

        print("total voxels:", voxel_cnt)
        print("min/max:", masked_intensities_erosion_out.min(), masked_intensities_erosion_out.max())
        print("p99, p99.9, p99.99:", np.percentile(masked_intensities_erosion_out, [99, 99.9, 99.99]))
        print("count > 3000:", np.sum(masked_intensities_erosion_out > 3000), " / ", masked_intensities_erosion_out.size)
        print("count > 5000:", np.sum(masked_intensities_erosion_out > 5000))

        # show_label_overlay(input_file, outputs_file_path + "/postproc_2d.nii.gz", z=None, alpha=0.4)

        # Print and save segment label statistics
        stats = []

        img = sitk.ReadImage(outputs_file_path + "/postproc_2d.nii.gz")
        # (z, y, x)
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation("RAS")

        img_ras = orienter.Execute(img)
        img_arry = sitk.GetArrayFromImage(img_ras)

        ct_sitk  = sitk.ReadImage(outputs_file_path + "/converted_image.nii.gz")
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation("RAS")

        img_ct = orienter.Execute(ct_sitk)
        ct_arry = sitk.GetArrayFromImage(img_ct)

        # Grid control (spacing/size/direction/origin)
        assert img_ct.GetSize() == img_ras.GetSize()
        assert img_ct.GetSpacing() == img_ras.GetSpacing()
        #assert img_ct.GetOrigin() == img_ras.GetOrigin()
        print(f'Origins: CT: {img_ct.GetOrigin()} POSTPRO:{img_ras.GetOrigin()}')
        assert img_ct.GetDirection() == img_ras.GetDirection()

        for u in np.unique(img_arry):
            if u == 0:
                continue
            vals = ct_arry[img_arry == u]
            stats.append({
                "label": int(u),
                "mean": float(vals.mean()),
                "std": float(vals.std()),
                "min": float(vals.min()),
                "max": float(vals.max()),
            })

        print(f"{outputs_file_path} Cluster statistics:")

        for s in stats:
            print(f"Label {s['label']}: mean={s['mean']:.2f}, std={s['std']:.2f}, min={s['min']:.2f}, max={s['max']:.2f}")

        with open(outputs_file_path + "/" + "stats.txt", "w") as f:
            for s in stats:
                f.write(
                    f"label: {s['label']}, "
                    f"mean: {s['mean']:.4f}, "
                    f"std: {s['std']:.4f}, "
                    f"min: {s['min']:.4f}, "
                    f"max: {s['max']:.4f}\n"
                )
        with open(outputs_file_path + "/" + "voxel_sizes.txt", "w") as f:
            for i, item in enumerate(mask_list):
                f.write(f"Voxel Size of Segment {i}: {item}\n")