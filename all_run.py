import os
import torch
import pydicom as pyd
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
# from k_means_constrained import KMeansConstrained
from collections import Counter
from totalsegmentator.python_api import totalsegmentator
from totalsegmentator.map_to_binary import class_map
from skimage.morphology import binary_erosion, disk
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans
from scipy.ndimage import binary_dilation
import warnings

print('CUDA available:', torch.cuda.is_available())
torch.version.cuda
print(torch.__version__)

# --- utils for visualization ---
#
#
def resample_to_ref(moving, ref, is_label=True):
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(
        moving,                  # moving image
        ref,                     # reference image
        sitk.Transform(),        # identity transform
        interp,
        0,                       # default pixel value
        moving.GetPixelID()      # output pixel type
    )

def show_overlay(img_path, seg_path, z=None, alpha=0.35, contour=True):
    img = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(seg_path)

    # Grid control (spacing/size/direction/origin)
    same_grid = (img.GetSize() == seg.GetSize() and
                 img.GetSpacing() == seg.GetSpacing() and
                 img.GetOrigin() == seg.GetOrigin() and
                 img.GetDirection() == seg.GetDirection())
    if not same_grid:
        seg = resample_to_ref(seg, img, is_label=True)

    img_arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (z,y,x)
    seg_arr = sitk.GetArrayFromImage(seg)                      # (z,y,x)

    # Slice
    if z is None:
        # automatic
        counts = (seg_arr > 0).sum(axis=(1,2))
        z = int(np.argmax(counts)) if counts.max() > 0 else img_arr.shape[0]//2

    I = img_arr[z]
    S = seg_arr[z]

    # Normalization
    vmin, vmax = np.percentile(I, (1, 99))
    I_show = np.clip(I, vmin, vmax)

    plt.figure(figsize=(7,7))
    plt.imshow(I_show, cmap="gray")
    plt.title(f"z={z}")

    # 1) Alpha overlay
    mask = (S > 0)
    overlay = np.zeros((*S.shape, 4), dtype=np.float32)
    overlay[mask] = [1.0, 0.0, 0.0, alpha]
    plt.imshow(overlay)

    # 2) Contour
    if contour:
        plt.contour(mask.astype(np.uint8), levels=[0.5], linewidths=1)

    plt.axis("off")
    plt.show()

def show_label_overlay(img_path, seg_path, z=None, alpha=0.45):
    img = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(seg_path)

    if (img.GetSize(), img.GetSpacing(), img.GetOrigin(), img.GetDirection()) != \
       (seg.GetSize(), seg.GetSpacing(), seg.GetOrigin(), seg.GetDirection()):
        seg = resample_to_ref(seg, img, is_label=True)

    img_arr = sitk.GetArrayFromImage(img).astype(np.float32)
    seg_arr = sitk.GetArrayFromImage(seg).astype(np.int32)

    if z is None:
        counts = (seg_arr > 0).sum(axis=(1,2))
        z = int(np.argmax(counts)) if counts.max() > 0 else img_arr.shape[0]//2

    I = img_arr[z]
    L = seg_arr[z]

    vmin, vmax = np.percentile(I, (1, 99))
    I_show = np.clip(I, vmin, vmax)

    plt.figure(figsize=(7,7))
    plt.imshow(I_show, cmap="gray")

    # label color plot
    masked = np.ma.masked_where(L == 0, L)
    plt.imshow(masked, alpha=alpha)

    plt.title(f"z={z}")
    plt.axis("off")
    plt.show()

def list_dicom_files(root_dir):
    dicoms = []
    for dp, _, fns in os.walk(root_dir):
        data_file_path = dp
        for fn in fns:
            p = os.path.join(dp, fn)
            dicoms.append(p)
    return dicoms, data_file_path

def separate_epi_endo(mask_eroded, roi_lv_dilated, dilate_lv_extra=1):
    lv = roi_lv_dilated.copy()
    if dilate_lv_extra > 0:
        struct = np.ones((3,3,3), dtype=bool)
        for _ in range(dilate_lv_extra):
            lv = binary_dilation(lv, structure=struct)

    # Endo: mask LV neighborhood voxels
    mask_boundary = mask_eroded & binary_dilation(~mask_eroded, structure=np.ones((3,3,3), dtype=bool))
    endo = mask_boundary & binary_dilation(lv, structure=np.ones((3,3,3), dtype=bool))

    # Epi: inside the mask neigborhood outside voxels, LV side
    epi = mask_eroded & binary_dilation(~mask_eroded, structure=np.ones((3,3,3), dtype=bool)) & ~binary_dilation(lv, structure=np.ones((3,3,3), dtype=bool))

    return endo, epi

def solve_laplace(myocardium, endo, epi, max_iter=5000, tol=1e-5):

    coords = np.where(myocardium)
    z1 = max(0, coords[0].min() - 1)
    z2 = min(myocardium.shape[0], coords[0].max() + 2)
    
    y1 = max(0, coords[1].min() - 1)
    y2 = min(myocardium.shape[1], coords[1].max() + 2)
    
    x1 = max(0, coords[2].min() - 1)
    x2 = min(myocardium.shape[2], coords[2].max() + 2)

    print(f"Myo coords: x1y1z1: {x1},{y1},{z1} / x2y2z2: {x2},{y2},{z2}")

    myo = myocardium[z1:z2, y1:y2, x1:x2]
    en  = endo[z1:z2, y1:y2, x1:x2]
    ep  = epi[z1:z2, y1:y2, x1:x2]
    interior = myo & ~en & ~ep

    print(f"Mask shape: {myo.shape}")
    print(f"Mask sum: {np.sum(myo)}")
    print(f"Endo sum: {np.sum(en)}")
    print(f"Epi sum: {np.sum(ep)}")
    
    # Overlap controls
    print(f"Mask & Endo overlap: {np.sum(myo & en)}")
    print(f"Mask & Epi overlap: {np.sum(myo & ep)}")
    print(f"Endo & Epi overlap: {np.sum(en & ep)}")
    print(f"Interior sum: {np.sum(interior)}")

    # If interior is empty, return zeroes
    if np.sum(interior) == 0:
        warnings.warn("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!No interior points found in Laplace solver. Returning zeroes.!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    phi = np.zeros(myo.shape, dtype=np.float64)
    phi[ep] = 1.0
    phi[en] = 0.0

    interior_inner = interior[1:-1,1:-1,1:-1]

    for i in range(max_iter):
        phi_old = phi[interior].copy()

        avg = (phi[:-2,1:-1,1:-1] + phi[2:,1:-1,1:-1] +
            phi[1:-1,:-2,1:-1] + phi[1:-1,2:,1:-1] +
            phi[1:-1,1:-1,:-2] + phi[1:-1,1:-1,2:]) / 6.0

        phi[1:-1,1:-1,1:-1][interior_inner] = avg[interior_inner]

        delta = np.max(np.abs(phi[interior] - phi_old))
        if (i+1) % 100 == 0:
            print(f"Laplacian solver iter {i+1}, delta: {delta:.2e}")
        if delta < tol:
            print(f"converged at {i+1}")
            break

    phi_full = np.zeros(myocardium.shape, dtype=np.float64)
    phi_full[z1:z2, y1:y2, x1:x2] = phi
    return phi_full

def get_layers(phi, myocardium, n_layers=6):
    boundaries = np.linspace(0.0, 1.0, n_layers + 1)
    layer_map  = np.zeros(phi.shape, dtype=np.int16)
    for i in range(n_layers):
        lo, hi = boundaries[i], boundaries[i+1]
        mask = myocardium & (phi >= lo) & (phi < hi if i < n_layers-1 else phi <= hi)
        layer_map[mask] = i + 1
    return layer_map

def build_bc_label_volume(mask_eroded, endo_bc, epi_bc, dtype=np.uint8):
    mask_eroded = mask_eroded.astype(bool)
    endo_bc = endo_bc.astype(bool)
    epi_bc  = epi_bc.astype(bool)

    lab = np.zeros(mask_eroded.shape, dtype=dtype)
    
    lab[endo_bc] = 1
    lab[epi_bc]  = 2
    return lab

def get_angular_sectors(myocardium, roi_lv, n_sectors=18):
    # LV centroid
    coords = np.where(roi_lv)
    cy, cx = coords[1].mean(), coords[2].mean()  # Y and X, axial plane
    
    #ys, xs = np.where(myocardium.any(axis=0))
    
    myo_coords = np.where(myocardium)
    y = myo_coords[1] - cy
    x = myo_coords[2] - cx
    
    theta = np.arctan2(y, x)  # between -pi vs pi
    theta_norm = (theta + np.pi) / (2 * np.pi)  # 0-1
    
    sector_map = np.zeros(myocardium.shape, dtype=np.int16)
    boundaries = np.linspace(0, 1, n_sectors + 1)
    for i in range(n_sectors):
        mask = myocardium.copy()
        mask[myocardium] = (theta_norm >= boundaries[i]) & (theta_norm < boundaries[i+1])
        sector_map[mask] = i + 1
    
    return sector_map

def combine_sector_layer(sector_map, layer_map, n_sectors, n_layers):
    combined = np.zeros(sector_map.shape, dtype=np.int16)
    for s in range(1, n_sectors+1):
        for l in range(1, n_layers+1):
            combined[(sector_map == s) & (layer_map == l)] = (s-1)*n_layers + l
    return combined

def post_segmentation_processing(input_file, output_file, segmentation_mask = 1, erosion_on=1, r_mm=0.5, repeat_alg=2, clustering=3, seed_k=54, k_means_3d=1, n_z=18):
# Post Segmentation Methods
# clustering = 1 # 0 off, 1 cylindirical, 2 kmeans
#
    n_theta = int(seed_k / n_z)
    print(f"4CH/SAX sector number: {n_z} / {n_theta}")
    # read
    seg_sitk = sitk.ReadImage(output_file + ".nii")
    ct_sitk  = sitk.ReadImage(input_file)

    seg = sitk.GetArrayFromImage(seg_sitk)       # (z,y,x)
    ct  = sitk.GetArrayFromImage(ct_sitk)        # (z,y,x)

    spacing = seg_sitk.GetSpacing()              # (x,y,z)
    origin  = seg_sitk.GetOrigin()
    direction = seg_sitk.GetDirection()

    # --- ROI mask (bool) ---
    if segmentation_mask:
        roi = (seg > 0) & (seg < 2)
    else:
        roi = (seg > 0)

    # --- 2D erosion slice-wise ---
    mask_eroded = roi.copy()

    if erosion_on:
        r_vox_x = int(round(r_mm / spacing[0]))
        r_vox_y = int(round(r_mm / spacing[1]))
        r_vox = max(1, int(round((r_vox_x + r_vox_y) / 2)))

        print(f"roi_shape: {roi.shape} r_mm: {r_mm}, spacing: {spacing}, r_vox_x: {r_vox_x}, r_vox_y: {r_vox_y}")
        print(f"erosion radius in voxels: {r_vox}")
        print(f"spacing: {spacing}")
        print(f"erosion radius in mm / repeat: {r_mm} / {repeat_alg}")

        footprint_disk = disk(r_vox)

        mask_eroded = np.zeros_like(roi, dtype=bool)

        for z in range(roi.shape[0]):
            temp_field = roi[z]        
            for _ in range(repeat_alg):
                temp_field = binary_erosion(temp_field, footprint=footprint_disk)

            mask_eroded[z] = temp_field
        erosion_out = mask_eroded
    else:
        erosion_out = roi
        
    # --- apply mask to CT for visualize (optional) ---
    ct_masked = np.where(mask_eroded, ct, 0)

    mask = mask_eroded.astype(bool)
    intens = ct[mask].reshape(-1, 1).astype(np.float32) 
    # --- features ---
    coords = np.argwhere(mask).astype(np.float32)      # (N,3)

    if clustering == 1:
        def cart_to_cylindrical(coords):
            cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
            x = coords[:, 0] - cx
            y = coords[:, 1] - cy
            z = coords[:, 2]
            
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)  # -pi to pi
            
            return np.stack([r, theta, z], axis=1)

        def cylindrical_grid_clustering_v2(coords, n_z=6, n_theta=9):
            cyl = cart_to_cylindrical(coords)
            theta, z = cyl[:, 1], cyl[:, 2]
            
            labels = np.full(len(coords), -1, dtype=int)
            
            # Divide z by quantiles
            z_bins = np.percentile(z, np.linspace(0, 100, n_z + 1))
            z_bins[-1] += 1e-10
            z_labels = np.digitize(z, z_bins) - 1
            
            # calculate theta bins for each z slice separately
            for z_idx in range(n_z):
                mask = z_labels == z_idx
                theta_slice = theta[mask]
                
                # divide theta slice into n_theta quantiles
                theta_bins = np.percentile(theta_slice, np.linspace(0, 100, n_theta + 1))
                theta_bins[-1] += 1e-10
                theta_labels = np.digitize(theta_slice, theta_bins) - 1
                
                labels[mask] = z_idx * n_theta + theta_labels
            
            return labels

        labels = cylindrical_grid_clustering_v2(coords, n_z=int(n_z), n_theta=int(n_theta))

        counts = Counter(labels)
        print(f"Min: {min(counts.values())}, Max: {max(counts.values())}, Std: {np.std(list(counts.values())):.1f}")

        clustered = np.zeros(mask.shape, dtype=np.uint16)
        clustered[mask] = labels + 1       # 1..k
        out_arr = clustered

    elif clustering == 2:
        if k_means_3d:
            intens = ct[mask].reshape(-1, 1).astype(np.float32)  # (N,1)

            # normalize
            coords = (coords - coords.mean(0, keepdims=True)) / (coords.std(0, keepdims=True) + 1e-8)
            #intens = (intens - intens.mean(0, keepdims=True)) / (intens.std(0, keepdims=True) + 1e-8)

            # combine features
            X = np.hstack([coords])

            # --- kmeans ---
            kmeans = KMeans(n_clusters=seed_k, n_init="auto", random_state=0)
            labels = kmeans.fit_predict(X)     # 0..k-1

            clustered = np.zeros(mask.shape, dtype=np.uint16)
            clustered[mask] = labels + 1       # 1..k

            out_arr = clustered
        else:
            clustered = np.zeros_like(seg, dtype=np.uint16)  # (z,y,x)
            min_vox = 1
            for z in range(mask_eroded.shape[0]):
                m = mask_eroded[z]
                if m.sum() < max(min_vox, seed_k):
                    continue

                intens = ct[z][m].reshape(-1, 1).astype(np.float32)

                # normalize
                coords = (coords - coords.mean(0, keepdims=True)) / (coords.std(0, keepdims=True) + 1e-8)
                intens = (intens - intens.mean(0, keepdims=True)) / (intens.std(0, keepdims=True) + 1e-8)

                # feature combine (weightning)
                X = np.hstack([coords])  # (N, 1+2)

                kmeans = KMeans(n_clusters=seed_k, n_init="auto", random_state=0)
                lab = kmeans.fit_predict(X)  # 0..k-1

                clustered[z][m] = (lab + 1).astype(np.uint16)  # 1..k

            out_arr = clustered
    elif clustering == 3:
        dilate_voxel = r_vox*repeat_alg
        roi_lv = (seg == 3)
        roi_lv_dilated = binary_dilation(roi_lv, iterations=dilate_voxel)

        endo, epi = separate_epi_endo(mask_eroded, roi_lv_dilated, dilate_lv_extra=1)
        print("mask shape:", mask_eroded.shape, "endo_seed:", endo.sum(), "epi_seed:", epi.sum())
        #bc_labels = build_bc_label_volume(mask_eroded, endo, epi)

        phi = solve_laplace(mask_eroded, endo, epi, max_iter=1500, tol=1e-4)
        thickness_map = get_layers(phi, mask_eroded, n_layers=n_theta)

        sector_map = get_angular_sectors(mask_eroded, endo, n_sectors=n_z)

        out_arr = combine_sector_layer(sector_map, thickness_map, n_sectors=n_z, n_layers=n_theta) # n_layers for thickness SAX, n_sectors for angular 4CH. overall k = n_z * n_theta
    else:
        out_arr = mask_eroded.astype(np.uint16)

    return out_arr, erosion_out, spacing, origin, direction, intens, ct, mask_eroded, roi


if __name__ == "__main__":
    all_patients = os.listdir(r"/mnt/research/research/Data/MyocardialScarDefns_kycn/Data")
    #all_patients = ["p_068", "p_069"]
    print(f"All patient list: {all_patients}")
    output_save_path = r"/mnt/research/research/Data/MyocardialScarDefns_kycn/outs"
    base_root_dir = r"/mnt/research/research/Data/MyocardialScarDefns_kycn/Data"

    all_patients = ["p_040"]
    print(f"All patient list: {all_patients}")
    output_save_path = r"/mnt/research/research/Data/MyocardialScarDefns_kycn/outs"
    base_root_dir = r"/mnt/research/research/Data/MyocardialScarDefns_kycn/have_problems"

    if_folder_exist_overwrite = 1

    for p in all_patients:

        full_path = os.path.join(output_save_path, p)
        root_dir = os.path.join(base_root_dir, p)

        print("CHECK:", repr(full_path))  # debug

        if os.path.exists(full_path):
            if if_folder_exist_overwrite:
                outputs_file_path = os.path.join(output_save_path, p) + "/"
            else:
                print(f"Path and files exist, it will not work again! {os.path.join(output_save_path, p)}")
                continue
        else:
            os.makedirs(full_path, exist_ok=True)
            outputs_file_path = full_path + "/"

        print(f"Output file path: {outputs_file_path}")
        
        print(f"Processing patient: {root_dir}")
    
        dicom_files_name_list, data_file_path = list_dicom_files(root_dir)
        print(f"Data file path: {data_file_path}")
        # print(dicom_files_name_list)

        check_dcm = dicom_files_name_list[1]
        print(f"Checking DICOM file: {check_dcm}")
        dcm = pyd.dcmread(check_dcm)

        slope = float(dcm.RescaleSlope)
        intercept = float(dcm.RescaleIntercept)

        print(slope, intercept)

        raw = dcm.pixel_array.astype(np.int32)

        slope = float(dcm.RescaleSlope)
        intercept = float(dcm.RescaleIntercept)

        hu = raw * slope + intercept

        y, x = raw.shape[0]//2, raw.shape[1]//2
        print("raw:", raw[y, x], "hu_from_dcm:", hu[y, x])

        img = sitk.ReadImage(check_dcm)
        arr = sitk.GetArrayFromImage(img)       # (1, y, x) or (y, x)

        print(arr.shape, arr.max(), arr.min())
        print("sitk value:", arr[0,200,200] if arr.ndim==3 else arr[200,200])

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

        # For transform dcm to nifti, we can use sitk.GetArrayFromImage to get the 3D array and then save it as nifti using sitk.WriteImage. However, since we want to crop the image in z dimension, we can directly use sitk.RegionOfInterest to crop the image and then save it as nifti.
        #
        # 

        array = sitk.GetArrayFromImage(image)

        size = image.GetSize()  # (x,y,z)

        # (x_start, y_start, z_start)
        start = [0, 0, 0]

        # (x_size, y_size, z_size)
        z_size = 460 # we want to keep all slices, so we set z_size to the original size in z dimension like size[2]
        new_size = [size[0], size[1], size[2]]

        cropped = sitk.RegionOfInterest(image, new_size, start)

        sitk.WriteImage(cropped, outputs_file_path + "/converted_image.nii.gz")

        input_file = outputs_file_path + "/converted_image.nii.gz"
        output_file = outputs_file_path + "/segmentations"

        out_segmentation = totalsegmentator(
            input=input_file,
            output=output_file,
            fast=False, 
            preview = False,
            task="heartchambers_highres", 
            ml=True, 
            verbose=True
        )

        print(f"Solutions: {output_file}")

        class_map["heartchambers_highres"]

        out_arr, erosion_out, spacing, origin, direction, intens, ct, mask_eroded, roi = post_segmentation_processing(input_file, output_file, segmentation_mask = 1, erosion_on=1, r_mm=0.5, repeat_alg=2, clustering=3, seed_k=54, k_means_3d=1, n_z=18)

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

        out_sitk = sitk.GetImageFromArray(erosion_out.astype(np.uint16)) 
        out_sitk.SetSpacing(spacing)
        out_sitk.SetOrigin(origin)
        out_sitk.SetDirection(direction)

        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation('RAS')
        out_sitk = orienter.Execute(out_sitk)
        sitk.WriteImage(out_sitk, outputs_file_path + "/erosion_out.nii.gz")
        print("saved:", outputs_file_path + "/erosion_out.nii.gz")

        out_sitk = sitk.GetImageFromArray(roi.astype(np.uint16)) 
        out_sitk.SetSpacing(spacing)
        out_sitk.SetOrigin(origin)
        out_sitk.SetDirection(direction)

        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation('RAS')
        out_sitk = orienter.Execute(out_sitk)
        sitk.WriteImage(out_sitk, outputs_file_path + "/roi.nii.gz")
        print("saved:", outputs_file_path + "/roi.nii.gz")

        print(f'intens.shape: {ct.shape}')
        #plt.plot(intens)
        #plt.show

        print(f'One point HU check Array {array[array.shape[0]//2, array.shape[1]//2, 0]}')
        print(f'One point HU check CT {ct[ct.shape[0]//2, ct.shape[1]//2,0]}')

        masked_intensities_before_kmeans = ct[mask_eroded]

        voxel_cnt = np.prod(ct.shape)

        print("total voxels:", voxel_cnt)
        print("min/max:", masked_intensities_before_kmeans.min(), masked_intensities_before_kmeans.max())
        print("p99, p99.9, p99.99:", np.percentile(masked_intensities_before_kmeans, [99, 99.9, 99.99]))
        print("count > 3000:", np.sum(masked_intensities_before_kmeans > 3000), " / ", masked_intensities_before_kmeans.size)
        print("count > 5000:", np.sum(masked_intensities_before_kmeans > 5000))

        show_label_overlay(input_file, outputs_file_path + "/postproc_2d.nii.gz", z=None, alpha=0.4)

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