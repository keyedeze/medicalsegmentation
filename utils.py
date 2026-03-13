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
from scipy.ndimage import binary_dilation, distance_transform_edt
import warnings
import json
from sklearn.decomposition import PCA

args = {}

###### Visualization 
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

###### Dicom file listing
def list_dicom_files(root_dir):
    dicoms = []
    for dp, _, fns in os.walk(root_dir):
        data_file_path = dp
        for fn in fns:
            p = os.path.join(dp, fn)
            dicoms.append(p)
    return dicoms, data_file_path

###### For laplace solver
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

def solve_laplace_torch(myocardium, endo, epi, max_iter=5000, tol=1e-5, device='cuda'):
    # Check if CUDA is available
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Convert to torch tensors and move to device
    myocardium = torch.from_numpy(myocardium).bool().to(device)
    endo = torch.from_numpy(endo).bool().to(device)
    epi = torch.from_numpy(epi).bool().to(device)
    
    # Find bounding box
    coords = torch.where(myocardium)
    z1 = max(0, int(coords[0].min().item()) - 1)
    z2 = min(myocardium.shape[0], int(coords[0].max().item()) + 2)
    
    y1 = max(0, int(coords[1].min().item()) - 1)
    y2 = min(myocardium.shape[1], int(coords[1].max().item()) + 2)
    
    x1 = max(0, int(coords[2].min().item()) - 1)
    x2 = min(myocardium.shape[2], int(coords[2].max().item()) + 2)

    print(f"Myo coords: x1y1z1: {x1},{y1},{z1} / x2y2z2: {x2},{y2},{z2}")

    # Crop to bounding box
    myo = myocardium[z1:z2, y1:y2, x1:x2]
    en  = endo[z1:z2, y1:y2, x1:x2]
    ep  = epi[z1:z2, y1:y2, x1:x2]
    interior = myo & ~en & ~ep

    print(f"Mask shape: {myo.shape}")
    print(f"Mask sum: {int(myo.sum().item())}")
    print(f"Endo sum: {int(en.sum().item())}")
    print(f"Epi sum: {int(ep.sum().item())}")
    
    # Overlap controls
    print(f"Mask & Endo overlap: {int((myo & en).sum().item())}")
    print(f"Mask & Epi overlap: {int((myo & ep).sum().item())}")
    print(f"Endo & Epi overlap: {int((en & ep).sum().item())}")
    print(f"Interior sum: {int(interior.sum().item())}")

    # If interior is empty, return zeros
    if interior.sum() == 0:
        warnings.warn("No interior points found in Laplace solver. Returning zeros.")
        return torch.zeros(myocardium.shape, dtype=torch.float64).cpu().numpy()
    
    # Initialize phi
    phi = torch.zeros(myo.shape, dtype=torch.float32, device=device)  # float32 faster
    phi[ep] = 1.0
    phi[en] = 0.0

    interior_inner = interior[1:-1, 1:-1, 1:-1]

    # Iterative solver
    for i in range(max_iter):
        phi_old = phi[interior].clone()

        # 6-neighbor average
        avg = (phi[:-2, 1:-1, 1:-1] + phi[2:, 1:-1, 1:-1] +
               phi[1:-1, :-2, 1:-1] + phi[1:-1, 2:, 1:-1] +
               phi[1:-1, 1:-1, :-2] + phi[1:-1, 1:-1, 2:]) / 6.0

        phi[1:-1, 1:-1, 1:-1][interior_inner] = avg[interior_inner]

        # Convergence check
        delta = torch.max(torch.abs(phi[interior] - phi_old)).item()
        
        if (i+1) % 100 == 0:
            print(f"Laplacian solver iter {i+1}, delta: {delta:.2e}")
        
        if delta < tol:
            print(f"Converged at iteration {i+1}")
            break

    # Place back into full volume
    phi_full = torch.zeros(myocardium.shape, dtype=torch.float32, device=device)
    phi_full[z1:z2, y1:y2, x1:x2] = phi
    
    # Convert back to numpy and CPU
    return phi_full.cpu().numpy().astype('float64')

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

def get_layers(phi, myocardium, n_layers=6, mode="uniform", center=0.5, width=0.1):
    layer_map = np.zeros(phi.shape, dtype=np.int16)

    if mode == "uniform":
        boundaries = np.linspace(0.0, 1.0, n_layers + 1)
        for i in range(n_layers):
            lo, hi = boundaries[i], boundaries[i+1]
            mask = myocardium & (phi >= lo) & (phi < hi if i < n_layers-1 else phi <= hi)
            layer_map[mask] = i + 1

    elif mode == "band":
        lo = max(0.0, center - width / 2.0)
        hi = min(1.0, center + width / 2.0)
        mask = myocardium & (phi >= lo) & (phi <= hi)
        layer_map[mask] = 1

    return layer_map

def get_transmural_band_label(phi, myocardium, center=0.5, width=0.15, label=1):
    lo = max(0.0, center - width / 2.0)
    hi = min(1.0, center + width / 2.0)

    out = np.zeros(phi.shape, dtype=np.uint8)
    mask = myocardium & (phi >= lo) & (phi <= hi)
    out[mask] = label
    return out

def build_bc_label_volume(mask_eroded, endo_bc, epi_bc, dtype=np.uint8):
    mask_eroded = mask_eroded.astype(bool)
    endo_bc = endo_bc.astype(bool)
    epi_bc  = epi_bc.astype(bool)

    lab = np.zeros(mask_eroded.shape, dtype=dtype)
    
    lab[endo_bc] = 1
    lab[epi_bc]  = 2
    return lab

def get_angular_sectors(myocardium, roi_lv, n_sectors=18, plane='axial'):
    """
    plane: 'axial' (Y-X, short-axis, 4CH view)
           'sagittal' (Z-Y)
           'coronal' (Z-X)
    """
    coords = np.where(roi_lv)
    
    if plane == 'axial':
        c1, c2 = coords[1].mean(), coords[2].mean()  # Y, X
        dim1, dim2 = 1, 2
    elif plane == 'sagittal':
        c1, c2 = coords[0].mean(), coords[1].mean()  # Z, Y
        dim1, dim2 = 0, 1
    elif plane == 'coronal':
        c1, c2 = coords[0].mean(), coords[2].mean()  # Z, X
        dim1, dim2 = 0, 2
    else:
        raise ValueError("plane must be 'axial', 'sagittal', or 'coronal'")
    
    myo_coords = np.where(myocardium)
    coord1 = myo_coords[dim1] - c1
    coord2 = myo_coords[dim2] - c2
    
    theta = np.arctan2(coord1, coord2)
    theta_norm = (theta + np.pi) / (2 * np.pi)
    
    sector_map = np.zeros(myocardium.shape, dtype=np.int16)
    boundaries = np.linspace(0, 1, n_sectors + 1)
    for i in range(n_sectors):
        mask = myocardium.copy()
        mask[myocardium] = (theta_norm >= boundaries[i]) & (theta_norm < boundaries[i+1])
        sector_map[mask] = i + 1
    
    return sector_map

def combine_sector_layer_fast(sector_map, layer_map, mask, n_layers):
    out = np.zeros_like(sector_map, dtype=np.uint16)
    s = sector_map[mask].astype(np.int32)
    l = layer_map[mask].astype(np.int32)
    valid = (s > 0) & (l > 0)
    out_flat = out.reshape(-1)
    idx = np.flatnonzero(mask)[valid]
    out_flat[idx] = ((s[valid]-1) * n_layers + l[valid]).astype(np.uint16)
    return out

def combine_sector_layer(sector_map, layer_map, n_sectors, n_layers):
    combined = np.zeros(sector_map.shape, dtype=np.int16)
    for s in range(1, n_sectors+1):
        for l in range(1, n_layers+1):
            combined[(sector_map == s) & (layer_map == l)] = (s-1)*n_layers + l
    return combined

def post_segmentation_processing(image_file, segmentation_file, segmentation_mask = 1,  
                                 erosion_on=1, r_mm=0.5, repeat_alg=2, clustering=3, seed_k=54, k_means_3d=1, n_z=18, laplace_max_iter = 5000, laplace_tolerance = 1e-5):
    """
    outs:
    out_arr: output of the post segmentation processing 
    erosion_out: output of the erosion algorithm, if erosion is not used, it is the same as roi 
    spacing, origin, direction: pysical dimensions and directions of the image
    ct: input image nifti
    roi: region of interest, includes just chosen segmentation labels
    """
    # Post Segmentation Methods
    # clustering = 1 # 0 off, 1 cylindirical, 2 kmeans
    #
    n_theta = int(seed_k / n_z)
    print(f"4CH/SAX sector number: {n_z} / {n_theta}")
    # read
    seg_sitk = sitk.ReadImage(segmentation_file + ".nii")
    ct_sitk  = sitk.ReadImage(image_file)

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
            intens = (intens - intens.mean(0, keepdims=True)) / (intens.std(0, keepdims=True) + 1e-8)

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

        coords_all = np.argwhere(mask_eroded).astype(np.float32)
        # take center on LV/mask
        c = np.argwhere(mask_eroded).mean(axis=0)   # (z,y,x)
        #cy, cx = c[1], c[2]

        # 1) KMeans on surface (epi/endo)
        surface = epi  # or endo
        coords_s = np.argwhere(surface).astype(np.float32)

        mu = coords_s.mean(axis=0, keepdims=True)
        sig = coords_s.std(axis=0, keepdims=True) + 1e-8
        coords_s_n = (coords_s - mu) / sig

        kmeans = KMeans(n_clusters=n_z, n_init="auto", random_state=0).fit(coords_s_n)
        labels_s = kmeans.labels_  # only for surface

        sector0 = np.zeros(mask_eroded.shape, dtype=np.uint16)
        sector0[surface] = labels_s.astype(np.uint16) + 1
        counts = np.bincount(labels_s, minlength=n_z)

        print("KMeans clusters:", n_z)
        print("Cluster size min:", counts.min())
        print("Cluster size mean:", counts.mean())
        print("Cluster size max:", counts.max())
        print("Empty clusters:", np.sum(counts == 0))

        # 4) Propagate surface labels to entire mask using nearest-surface voxel
        # distance_transform_edt: calculate distance from 0 places -> surface==True points are 0
        _, inds = distance_transform_edt(~surface, return_indices=True)
        iz, iy, ix = inds

        sector_map_3d = np.zeros_like(sector0)
        sector_map_3d[mask_eroded] = sector0[iz, iy, ix][mask_eroded] 

        u = np.unique(sector_map_3d[mask_eroded])
        print("Unique sectors in volume:", len(u), "min/max:", u.min(), u.max())
        print("Any zero inside mask?:", np.any(sector_map_3d[mask_eroded] == 0))
        
        phi = solve_laplace_torch(mask_eroded, endo, epi, max_iter=laplace_max_iter, tol=laplace_tolerance)

        thickness_map = get_layers(phi, mask_eroded, n_layers=n_theta)
        print(f"Laplacian thickness sector map sector size: {n_theta}")
        u, c = np.unique(thickness_map[mask_eroded], return_counts=True)

        print("Thickness layers:", u)
        print("Layer voxel counts:", dict(zip(u, c)))

        #sector_map = get_angular_sectors(mask_eroded, endo, n_sectors=n_z, plane='axial')

        out_arr = combine_sector_layer_fast(sector_map_3d, thickness_map, mask_eroded, n_layers=n_theta) # n_layers for thickness SAX, n_sectors for angular 4CH. overall k = n_z * n_theta

        u_sector, c_sector = np.unique(out_arr, return_counts=True)
        print("Sector count:", len(u_sector))
        print("Sector min id:", u_sector.min(), "max id:", u_sector.max())
        print("Sector voxel min:", c_sector.min())
        print("Sector voxel mean:", c_sector.mean())
        print("Sector voxel max:", c_sector.max())
        print(f"Total sector size: {len(np.unique(out_arr))-1}")

    else:
        out_arr = mask_eroded.astype(np.uint16)

    return out_arr, erosion_out, spacing, origin, direction, ct, roi

def read_write_nifti(
    write_mode,
    path,
    file_name,
    array_for_writing=None,
    reference_img=None,
    output_orientation="RAS"
):
    full_path = os.path.join(path, file_name)

    if write_mode:

        if array_for_writing is None:
            raise ValueError("array_for_writing needs for writing operation!!")

        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation(output_orientation)

        out_sitk = sitk.GetImageFromArray(array_for_writing)

        if reference_img is not None:
            out_sitk.CopyInformation(reference_img)

        out_sitk = orienter.Execute(out_sitk)

        sitk.WriteImage(out_sitk, full_path)

        print("Saved:", full_path)
        print(f"Size: {out_sitk.GetSize()}")
        print(f"Spacing: {out_sitk.GetSpacing()}")
        print(f"Origin: {out_sitk.GetOrigin()}")
        print(f"Direction: {out_sitk.GetDirection()}")

        return out_sitk

    else:

        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation(output_orientation)

        img = sitk.ReadImage(full_path)
        img_oriented = orienter.Execute(img)

        print("Loaded:", full_path)
        print(f"Size: {img_oriented.GetSize()}")
        print(f"Spacing: {img_oriented.GetSpacing()}")
        print(f"Origin: {img_oriented.GetOrigin()}")
        print(f"Direction: {img_oriented.GetDirection()}")

        return img_oriented

def make_erosion(roi, spacing, r_mm, repeat_alg):
    """
    roi     : NumPy array, shape = (z, y, x)
    spacing : tuple/list, (sx, sy, sz) veya SimpleITK'den gelen spacing
    r_mm    : erosion yarıçapı (mm)
    repeat_alg : erosion tekrar sayısı
    """

    sx, sy = spacing[0], spacing[1]

    r_vox_x = int(round(r_mm / sx))
    r_vox_y = int(round(r_mm / sy))
    r_vox = max(1, int(round((r_vox_x + r_vox_y) / 2)))

    print(f"roi_shape: {roi.shape}")
    print(f"r_mm: {r_mm}")
    print(f"spacing: {spacing}")
    print(f"r_vox_x: {r_vox_x}, r_vox_y: {r_vox_y}")
    print(f"erosion radius in voxels: {r_vox}")
    print(f"erosion radius in mm / repeat: {r_mm} / {repeat_alg}")

    footprint_disk = disk(r_vox)
    mask_eroded = np.zeros_like(roi, dtype=bool)

    for z in range(roi.shape[0]):
        temp_field = roi[z].astype(bool)

        for _ in range(repeat_alg):
            temp_field = binary_erosion(temp_field, footprint=footprint_disk)

        mask_eroded[z] = temp_field
    roi_eroded = roi * mask_eroded
    print(f"Erosion finished")
    return roi_eroded, r_vox

def hu_heatmap_export(
volume_path,
seg_path,
output_scalar_path,
output_rgb_path,
n_levels=10,
seg_labels=None,
colormap="turbo",
outside_value_scalar=-9999.0,
outside_value_rgb=(0, 0, 0),
):
    """
    Produces:
        1) Scalar heatmap NIfTI  (float32)
        2) RGB heatmap NIfTI     (uint8, vector image)
        3) JSON scale file       (HU range, levels, colormap)

    Parameters
    ----------
    volume_path : str
        Input HU volume (.nii or .nii.gz)
    seg_path : str
        Input segmentation (.nii or .nii.gz)
    output_scalar_path : str
        Output scalar heatmap NIfTI path
    output_rgb_path : str
        Output RGB heatmap NIfTI path
    n_levels : int
        Number of quantization levels
    seg_labels : list or None
        If None -> seg > 0 used
        else -> only listed labels used
    colormap : str
        matplotlib colormap name: turbo, viridis, plasma, inferno, magma, jet...
    outside_value_scalar : float
        Value outside ROI in scalar heatmap
    outside_value_rgb : tuple
        RGB color outside ROI
    """

    # -----------------------------
    # 1) Load + canonicalize to RAS
    # -----------------------------
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation("RAS")

    vol_img = sitk.ReadImage(volume_path)
    seg_img = sitk.ReadImage(seg_path)

    vol_img = orienter.Execute(vol_img)
    seg_img = orienter.Execute(seg_img)

    vol = sitk.GetArrayFromImage(vol_img).astype(np.float32)   # (z,y,x)
    seg = sitk.GetArrayFromImage(seg_img)

    if vol.shape != seg.shape:
        raise ValueError(f"Volume and segmentation not matched: {vol.shape} vs {seg.shape}")

    # -----------------------------
    # 2) Build mask
    # -----------------------------
    if seg_labels is None:
        mask = seg > 0
    else:
        mask = np.isin(seg, seg_labels)

    if not np.any(mask):
        raise ValueError("Mask empty")

    # -----------------------------
    # 3) HU range in ROI
    # -----------------------------
    hu_vals = vol[mask]
    hu_min = float(hu_vals.min())
    hu_max = float(hu_vals.max())

    print("Segment HU min:", hu_min)
    print("Segment HU max:", hu_max)

    # -----------------------------
    # 4) Quantized scalar heatmap
    # -----------------------------
    scalar_heatmap = np.full_like(vol, outside_value_scalar, dtype=np.float32)

    if hu_max == hu_min:
        levels = np.array([hu_min], dtype=np.float32)
        scalar_heatmap[mask] = hu_min
    else:
        levels = np.linspace(hu_min, hu_max, n_levels, dtype=np.float32)
        idx = np.abs(vol[mask][..., None] - levels).argmin(axis=-1)
        scalar_heatmap[mask] = levels[idx]

    # -----------------------------
    # 5) RGB heatmap from scalar
    # -----------------------------
    rgb = np.zeros((*scalar_heatmap.shape, 3), dtype=np.uint8)

    if hu_max == hu_min:
        norm = np.zeros_like(scalar_heatmap, dtype=np.float32)
        norm[mask] = 1.0
    else:
        norm = np.zeros_like(scalar_heatmap, dtype=np.float32)
        norm[mask] = (scalar_heatmap[mask] - hu_min) / (hu_max - hu_min)
        norm = np.clip(norm, 0, 1)

    cmap = plt.get_cmap(colormap)
    rgb_float = cmap(norm)[..., :3]  # values in [0,1]
    rgb = (rgb_float * 255).astype(np.uint8)

    outside_value_rgb = np.array(outside_value_rgb, dtype=np.uint8)
    rgb[~mask] = outside_value_rgb

    # -----------------------------
    # 6) Save scalar NIfTI
    # -----------------------------
    scalar_img = sitk.GetImageFromArray(scalar_heatmap.astype(np.float32))
    scalar_img.CopyInformation(vol_img)
    sitk.WriteImage(scalar_img, output_scalar_path)

    # -----------------------------
    # 7) Save RGB NIfTI
    # -----------------------------
    rgb_img = sitk.GetImageFromArray(rgb, isVector=True)
    rgb_img.CopyInformation(vol_img)
    sitk.WriteImage(rgb_img, output_rgb_path)

    # -----------------------------
    # 8) Save scale metadata JSON
    # -----------------------------
    json_path = output_rgb_path
    if json_path.endswith(".nii.gz"):
        json_path = json_path[:-7] + "_scale.json"
    else:
        json_path = os.path.splitext(json_path)[0] + "_scale.json"

    scale_info = {
        "HU_min": hu_min,
        "HU_max": hu_max,
        "n_levels": int(n_levels),
        "levels": [float(x) for x in levels.tolist()],
        "colormap": colormap,
        "orientation": "RAS",
        "scalar_output": output_scalar_path,
        "rgb_output": output_rgb_path,
    }

    with open(json_path, "w") as f:
        json.dump(scale_info, f, indent=2)

    # -----------------------------
    # 9) Info
    # -----------------------------
    print("Scalar heatmap saved :", output_scalar_path)
    print("RGB heatmap saved    :", output_rgb_path)
    print("Scale JSON saved     :", json_path)


"""
AHA 17-Segment Myocardial Segmentation
Robust AHA17 segmentation using anatomical masks.

Mask mapping:
    1: heart_myocardium
    2: heart_atrium_left
    3: heart_ventricle_left
    4: heart_atrium_right
    5: heart_ventricle_right
    6: aorta
    7: pulmonary_artery

Method:
  - Long axis      : PCA of the myocardium
  - Axis direction : Aorta voxels closest to the LV (aortic valve = BASE)
                     → consistent across patients, independent of aorta size
  - Apex detection : The most distal 1% of myocardium voxels (small t)
  - Base detection : 90th percentile of LV voxels along the base direction (large t)
  - z_norm         : t_apex → 0, t_base → 1 (single reference system)
  - RV insertion   : Intersection of RV and myocardium → angular zero reference
  - AHA segment order: segment 1 = anterior-septal (CCW from RV insertion)
"""

# ---------------------------------------------------------------------------
# Long axis detection
# ---------------------------------------------------------------------------
def compute_long_axis(
    myocardium_mask: np.ndarray,
    lv_mask: np.ndarray,
    aorta_mask: np.ndarray,
):
    """
    Computes the long axis vector and projection values.

    Axis direction: from APEX to BASE
      - The aortic valve region is estimated from the aorta voxels closest to the LV
      - Independent of the absolute size of the aorta

    Returns
    -------
    axis   : unit vector pointing from APEX to BASE
    center : center of mass of myocardium voxels
    pts    : myocardium voxel coordinates (N, 3)
    t_vals : projection value for each voxel (N,)
    """

    pts    = np.argwhere(myocardium_mask).astype(np.float64)
    center = pts.mean(axis=0)

    pca = PCA(n_components=3)
    pca.fit(pts)
    axis = pca.components_[0]

    if aorta_mask.any() and lv_mask.any():
        # Distance map to LV
        lv_dist = distance_transform_edt(~lv_mask)

        # Distances of aorta voxels to the LV
        aorta_idx   = np.argwhere(aorta_mask)
        aorta_dists = lv_dist[aorta_idx[:, 0], aorta_idx[:, 1], aorta_idx[:, 2]]

        # Closest 5% of aorta voxels to the LV = aortic valve region = BASE reference
        cutoff         = np.percentile(aorta_dists, 5)
        base_ref       = aorta_idx[aorta_dists <= cutoff].astype(np.float64).mean(axis=0)

        # Align axis toward the BASE reference point (APEX → BASE)
        if np.dot(axis, base_ref - center) < 0:
            axis = -axis

        print(f"  Axis direction: LV-nearest aorta reference (APEX→BASE)")
        print(f"  BASE reference point (z,y,x): {base_ref.round(1)}")
    else:
        # Fallback: opposite direction of the LV centroid = BASE
        lv_center = np.argwhere(lv_mask).astype(np.float64).mean(axis=0)
        if np.dot(axis, lv_center - center) > 0:
            axis = -axis
        print(f"  Axis direction: LV centroid fallback")

    t_vals = (pts - center) @ axis
    return axis, center, pts, t_vals


# ---------------------------------------------------------------------------
# Anatomical boundary detection
# ---------------------------------------------------------------------------
def compute_anatomical_boundaries(
    lv_mask: np.ndarray,
    axis: np.ndarray,
    center: np.ndarray,
    myo_pts: np.ndarray,
) -> dict:
    """
    Automatically computes anatomical boundaries using myocardium and LV masks.

    Axis orientation: APEX → BASE
      apex = small t  → myocardium percentile(1)
      base = large t  → LV percentile(90)

    Returns
    -------
    t_apex_abs      : apex projection value
    t_base_abs      : basal plane projection value
    lv_length_vox   : long axis length
    apex_fraction   : apex cap fraction (~0.08)
    apical_ring_end : end of apical ring (~0.33)
    mid_ring_end    : end of mid ring (~0.67)
    """

    # Apex = distal 1% of myocardium voxels (small t)
    t_myo      = (myo_pts - center) @ axis
    t_apex_abs = np.percentile(t_myo, 1)

    # Base = upper 10% of LV voxels toward base (large t)
    lv_pts     = np.argwhere(lv_mask).astype(np.float64)
    t_lv       = (lv_pts - center) @ axis
    t_base_abs = np.percentile(t_lv, 90)

    print(f"  t_apex (myo %1)           : {t_apex_abs:.1f}")
    print(f"  t_base (LV %90)           : {t_base_abs:.1f}")

    lv_length = t_base_abs - t_apex_abs
    if lv_length <= 0:
        raise ValueError(
            f"Invalid LV length: t_apex={t_apex_abs:.2f}, t_base={t_base_abs:.2f}\n"
            f"  Myo t: [{t_myo.min():.1f}, {t_myo.max():.1f}]\n"
            f"  LV  t: [{t_lv.min():.1f},  {t_lv.max():.1f}]\n"
            "  The axis direction may still be inverted."
        )

    third = 1.0 / 3.0
    return {
        "t_apex_abs":      t_apex_abs,
        "t_base_abs":      t_base_abs,
        "lv_length_vox":   lv_length,
        "apex_fraction":   0.08,
        "apical_ring_end": third,       # 0.33
        "mid_ring_end":    third * 2,   # 0.67
    }

# ---------------------------------------------------------------------------
# RV insertion point detection
# ---------------------------------------------------------------------------
def find_rv_insertion_point(myocardium_mask: np.ndarray, rv_mask: np.ndarray):
    """
    Finds the centroid of the intersection region between the RV and myocardium.

    Returns
    -------
    [z, y, x] array or None
    """
    myo_dilated  = binary_dilation(myocardium_mask, iterations=3)
    rv_dilated   = binary_dilation(rv_mask,         iterations=3)
    intersection = myo_dilated & rv_dilated

    if not intersection.any():
        intersection = myocardium_mask & rv_mask
    if not intersection.any():
        return None

    pts = np.argwhere(intersection).astype(np.float64)
    return pts.mean(axis=0)


# ---------------------------------------------------------------------------
# AHA 17-segment
# ---------------------------------------------------------------------------
def aha17_segment(
    segmentation_array: np.ndarray,
    label_map: dict = None,
    boundaries: dict = None,
) -> np.ndarray:
    """
    Produce AHA 17 segment label map with using anatomical masks

    segmentation_array : 3D numpy array (z, y, x), integer
    boundaries         : compute_anatomical_boundaries() output,
                         if None, will be calculate inside

    out
    --------
    labels : 3D numpy array, dtype=int32
             1-17 AHA segment numbers (out of myocardium = 0)

    AHA 17-segment
    ---------------------
    Basal  (z_norm >= mid_ring_end)                → seg  1- 6  (60° x 6)
    Mid    (apical_ring_end <= z < mid_ring_end)   → seg  7-12  (60° x 6)
    Apical (apex_fraction   <= z < apical_ring_end)→ seg 13-16  (90° x 4)
    Apex   (z < apex_fraction)                     → seg 17

    (from RV insertion CCW):
      Seg 1 : Anteroseptal   (  0°– 60°)
      Seg 2 : Anterior       ( 60°–120°)
      Seg 3 : Anterolateral  (120°–180°)
      Seg 4 : Inferolateral  (180°–240°)
      Seg 5 : Inferior       (240°–300°)
      Seg 6 : Inferoseptal   (300°–360°)
    """
    if label_map is None:
        label_map = {
            1: 'heart_myocardium',
            2: 'heart_atrium_left',
            3: 'heart_ventricle_left',
            4: 'heart_atrium_right',
            5: 'heart_ventricle_right',
            6: 'aorta',
            7: 'pulmonary_artery',
        }

    inv_map = {v: k for k, v in label_map.items()}

    myocardium_mask = (segmentation_array == inv_map['heart_myocardium'])
    lv_mask         = (segmentation_array == inv_map['heart_ventricle_left'])
    rv_mask         = (segmentation_array == inv_map['heart_ventricle_right'])
    aorta_mask      = (segmentation_array == inv_map['aorta'])

    if not myocardium_mask.any():
        raise ValueError("Myocardium mask empty!")
    if not lv_mask.any():
        raise ValueError("LV ventricle mask empty!")

    print(f"  Myocardium voxel number   : {myocardium_mask.sum():,}")
    print(f"  LV ventricle voxel number : {lv_mask.sum():,}")
    print(f"  RV ventricle voxel number : {rv_mask.sum():,}")
    print(f"  Aorta voxel number        : {aorta_mask.sum():,}")

    # ------------------------------------------------------------------
    # 1) LAX, from APEX to BASE
    # ------------------------------------------------------------------
    axis, center, pts, t_vals = compute_long_axis(
        myocardium_mask, lv_mask, aorta_mask
    )
    print(f"  Long Axis (Array Space)  : {axis.round(3)}")

    # ------------------------------------------------------------------
    # 2) Anatomical boundaries
    # ------------------------------------------------------------------
    if boundaries is None:
        boundaries = compute_anatomical_boundaries(
            lv_mask, axis, center, pts
        )

    t_apex          = boundaries["t_apex_abs"]
    t_base          = boundaries["t_base_abs"]
    lv_length       = boundaries["lv_length_vox"]
    apex_fraction   = boundaries["apex_fraction"]
    apical_ring_end = boundaries["apical_ring_end"]
    mid_ring_end    = boundaries["mid_ring_end"]

    print(f"  LV length (voksel)        : {lv_length:.1f}")
    print(f"  apex_fraction             : {apex_fraction:.3f}")
    print(f"  apical_ring_end           : {apical_ring_end:.3f}")
    print(f"  mid_ring_end              : {mid_ring_end:.3f}")

    # ------------------------------------------------------------------
    # 3) z_norm: t_apex=0, t_base=1
    # ------------------------------------------------------------------
    z_norm = (t_vals - t_apex) / lv_length

    # ------------------------------------------------------------------
    # 4) Orthonormal basis for SAX
    # ------------------------------------------------------------------
    ref = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(axis, ref)) > 0.95:
        ref = np.array([1.0, 0.0, 0.0])
    v1 = ref - np.dot(ref, axis) * axis
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(axis, v1)
    v2 /= np.linalg.norm(v2)

    # ------------------------------------------------------------------
    # 5) RV insertion point (angular)
    # ------------------------------------------------------------------
    angle_offset = 0.0
    if rv_mask.any():
        rv_ins = find_rv_insertion_point(myocardium_mask, rv_mask)
        if rv_ins is not None:
            rel          = rv_ins - center
            rv_angle     = np.degrees(np.arctan2(rel @ v2, rel @ v1))
            angle_offset = (rv_angle - 60.0 + 360.0) % 360.0
            print(f"  RV insertion angle        : {rv_angle:.1f},  offset: {angle_offset:.1f}")
        else:
            print("  WARNING: Can not find RV insertion, angular offset = 0")
    else:
        print("  WARNING: There is no RV mask, angular offset = 0")

    # ------------------------------------------------------------------
    # 6) Segment ataması
    # ------------------------------------------------------------------
    labels = np.zeros(myocardium_mask.shape, dtype=np.int32)

    for i, p in enumerate(pts):
        rel   = p - center
        theta = (np.degrees(np.arctan2(rel @ v2, rel @ v1)) - angle_offset + 360.0) % 360.0
        z     = z_norm[i]

        if z < apex_fraction:
            seg = 17                                # apex cap
        elif z < apical_ring_end:
            seg = 12 + int(theta // 90.0) + 1      # 13–16
        elif z < mid_ring_end:
            seg = 6  + int(theta // 60.0) + 1      # 7–12
        else:
            seg = 0  + int(theta // 60.0) + 1      # 1–6

        seg = max(1, min(17, seg))
        labels[int(p[0]), int(p[1]), int(p[2])] = seg

    counts = {s: int((labels == s).sum()) for s in range(1, 18)}
    print("\n  Segment voxel numbers:")
    for s, c in counts.items():
        print(f"    Seg {s:2d}: {c:6,} voxel")

    empty = [s for s, c in counts.items() if c == 0]
    if empty:
        print(f"\n  WARNING: Empty Segments: {empty}")

    diag = {
        "axis":         axis,
        "center":       center,
        "v1":           v1,
        "v2":           v2,
        "t_apex":       t_apex,
        "t_base":       t_base,
        "lv_length":    lv_length,
        "angle_offset": angle_offset,
        "boundaries":   boundaries,
    }

    return labels, diag

# ---------------------------------------------------------------------------
# Subdivision
# ---------------------------------------------------------------------------
def subdivide_aha_segments(
    labels: np.ndarray,
    seg_array: np.ndarray,
    diag: dict,
    n_angular: int = 4,
    n_longitudinal: int = 4,
    label_map: dict = None,
) -> np.ndarray:
    """
    Subdivides each AHA segment into n_angular x n_longitudinal subsegments.
    Total: 17 x (n_angular * n_longitudinal) subsegments.

    Method:
      - Compute the true t and theta ranges for each segment
      - Longitudinal: divide the range t_min → t_max into n_longitudinal equal parts
      - Angular     : divide the range theta_min → theta_max into n_angular equal parts
      - Sub ID      : (aha_seg - 1) * n_sub + l_bin * n_angular + a_bin + 1
    """
    if label_map is None:
        label_map = {
            1: "heart_myocardium",
            2: "heart_atrium_left",
            3: "heart_ventricle_left",
            4: "heart_atrium_right",
            5: "heart_ventricle_right",
            6: "aorta",
            7: "pulmonary_artery",
        }

    inv_map  = {v: k for k, v in label_map.items()}
    axis     = diag["axis"]
    center   = diag["center"]
    v1       = diag["v1"]
    v2       = diag["v2"]

    myocardium_mask = (seg_array == inv_map["heart_myocardium"])

    n_sub = n_angular * n_longitudinal
    print(f"  {n_angular} angular x {n_longitudinal} longitudinal = {n_sub} total segment")

    # Compute true ranges for each AHA segment
    seg_info = {}
    for s in range(1, 18):
        seg_pts = np.argwhere(labels == s).astype(np.float64)
        if len(seg_pts) == 0:
            continue

        # Longitudinal range
        t_vals = (seg_pts - center) @ axis

        # Angular range: compute theta for each voxel
        rel      = seg_pts - center
        x_vals   = rel @ v1
        y_vals   = rel @ v2

        # Centroid angle
        theta_ctr = (np.degrees(np.arctan2(y_vals.mean(), x_vals.mean())) + 360.0) % 360.0

        # Delta relative to centroid (-180, +180)
        theta_all = (np.degrees(np.arctan2(y_vals, x_vals)) + 360.0) % 360.0
        delta_all = (theta_all - theta_ctr + 540.0) % 360.0 - 180.0

        seg_info[s] = {
            "t_min":     t_vals.min(),
            "t_max":     t_vals.max(),
            "theta_ctr": theta_ctr,
            "d_min":     delta_all.min(),   # true angular range of the segment
            "d_max":     delta_all.max(),
        }

    sub_labels = np.zeros(labels.shape, dtype=np.int32)
    pts        = np.argwhere(myocardium_mask).astype(np.float64)

    for p in pts:
        zi, yi, xi = int(p[0]), int(p[1]), int(p[2])
        s = labels[zi, yi, xi]
        if s == 0 or s not in seg_info:
            continue

        si = seg_info[s]

        # Longitudinal bin within the true t range of the segment
        t       = float((p - center) @ axis)
        t_width = si["t_max"] - si["t_min"]
        l_local = np.clip((t - si["t_min"]) / t_width, 0.0, 1.0 - 1e-9) if t_width > 0 else 0.0
        l_bin   = min(int(l_local * n_longitudinal), n_longitudinal - 1)

        # Angular bin within the true delta range of the segment
        rel     = p - center
        theta_p = (np.degrees(np.arctan2(rel @ v2, rel @ v1)) + 360.0) % 360.0
        delta   = (theta_p - si["theta_ctr"] + 540.0) % 360.0 - 180.0
        d_width = si["d_max"] - si["d_min"]
        a_local = np.clip((delta - si["d_min"]) / d_width, 0.0, 1.0 - 1e-9) if d_width > 0 else 0.0
        a_bin   = min(int(a_local * n_angular), n_angular - 1)

        sub_id = (s - 1) * n_sub + l_bin * n_angular + a_bin + 1
        sub_labels[zi, yi, xi] = sub_id

    # Statistics
    total_sub = 17 * n_sub
    counts    = {i: int((sub_labels == i).sum()) for i in range(1, total_sub + 1)}
    filled    = sum(1 for c in counts.values() if c > 0)
    empty_sub = [i for i, c in counts.items() if c == 0]
    print(f"  Filled subsegments: {filled} / {total_sub}")
    if empty_sub:
        print(f"  Empty subsegments : {empty_sub[:30]}{'...' if len(empty_sub) > 30 else ''}")

    return sub_labels