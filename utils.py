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
        surface = epi  # veya endo
        coords_s = np.argwhere(surface).astype(np.float32)

        mu = coords_s.mean(axis=0, keepdims=True)
        sig = coords_s.std(axis=0, keepdims=True) + 1e-8
        coords_s_n = (coords_s - mu) / sig

        kmeans = KMeans(n_clusters=n_z, n_init="auto", random_state=0).fit(coords_s_n)
        labels_s = kmeans.labels_  # sadece surface için

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