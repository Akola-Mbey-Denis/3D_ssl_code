import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple
from einops import rearrange

import numpy as np
import torch
import torch.nn.functional as F

from scipy.ndimage import map_coordinates
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

def draw_epipolar_line_between_cameras(
    pixel_cam1: Tuple[int, int],
    K1: np.ndarray,
    RT1_w2c: np.ndarray,
    img_cam2: np.ndarray,
    K2: np.ndarray,
    RT2_w2c: np.ndarray,
    depth_cam1: float = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Simple function: given a pixel in camera 1, draw the epipolar line in camera 2.
    If depth is provided, also draw the exact projected point.
    
    Args:
        pixel_cam1: (u, v) pixel coordinates in camera 1
        K1: 3x3 intrinsic matrix for camera 1
        RT1_w2c: 4x4 world-to-camera transform for camera 1
        img_cam2: Image from camera 2 to draw on
        K2: 3x3 intrinsic matrix for camera 2
        RT2_w2c: 4x4 world-to-camera transform for camera 2
        depth_cam1: Optional depth value at pixel in camera 1
        color: Line color (B, G, R)
        thickness: Line thickness
    
    Returns:
        img_cam2 with epipolar line drawn (and projected point if depth given)
    """
    img_out = img_cam2.copy()
    h, w = img_out.shape[:2]
    
    # Step 1: Compute Fundamental Matrix
    # Get relative transformation from cam1 to cam2
    RT1_c2w = np.linalg.inv(RT1_w2c)
    RT_1to2 = RT2_w2c @ RT1_c2w
    R = RT_1to2[:3, :3]
    t = RT_1to2[:3, 3]
    
    # Essential matrix: E = [t]_x @ R
    t_x = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_x @ R
    
    # Fundamental matrix: F = K2^-T @ E @ K1^-1
    F = np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)
    
    # Step 2: Compute epipolar line in camera 2
    p1 = np.array([pixel_cam1[0], pixel_cam1[1], 1.0])
    line2 = F @ p1  # line: [a, b, c] where ax + by + c = 0
    
    # Step 3: Draw the epipolar line
    a, b, c = line2
    
    # Find where line intersects image boundaries
    points = []
    
    if abs(b) > 1e-6:
        # Intersection with left edge (x=0)
        y = -c / b
        if 0 <= y <= h:
            points.append((0, int(y)))
        
        # Intersection with right edge (x=w)
        y = -(a * w + c) / b
        if 0 <= y <= h:
            points.append((w, int(y)))
    
    if abs(a) > 1e-6:
        # Intersection with top edge (y=0)
        x = -c / a
        if 0 <= x <= w:
            points.append((int(x), 0))
        
        # Intersection with bottom edge (y=h)
        x = -(b * h + c) / a
        if 0 <= x <= w:
            points.append((int(x), h))
    
    # Draw line through first two intersection points
    if len(points) >= 2:
        cv2.line(img_out, points[0], points[1], color, thickness)
    
    # Step 4: If depth is provided, compute exact projected point
    if depth_cam1 is not None:
        # Backproject pixel to 3D in camera 1
        p1_normalized = np.linalg.inv(K1) @ p1
        point_cam1 = p1_normalized * depth_cam1
        point_cam1_homog = np.append(point_cam1, 1.0)
        
        # Transform to world coordinates
        point_world = RT1_c2w @ point_cam1_homog
        
        # Transform to camera 2
        point_cam2 = RT2_w2c @ point_world
        point_cam2_3d = point_cam2[:3]
        
        # Project to camera 2 image plane
        if point_cam2_3d[2] > 0:  # Check if in front of camera
            pixel_cam2 = K2 @ point_cam2_3d
            pixel_cam2 = pixel_cam2[:2] / pixel_cam2[2]
            
            # Draw the projected point
            px, py = int(pixel_cam2[0]), int(pixel_cam2[1])
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(img_out, (px, py), 6, (0, 0, 255), -1)
                cv2.circle(img_out, (px, py), 8, (255, 255, 255), 2)
    
    return img_out


def visualize_epipolar_geometry(
    K1: np.ndarray = None,
    K2: np.ndarray = None,
    img1: np.ndarray = None,
    img2: np.ndarray = None,
    RT1_w2c: np.ndarray = None,
    RT2_w2c: np.ndarray = None,
    depth: float = 5.0,
):
    """
    Visualize epipolar geometry between two cameras.
    
    Args:
        pixel_cam1: Selected pixel in camera 1
        depth: Depth value for the 3D point
        img_size: (height, width) of images
        focal_length: Camera focal length in pixels
        rotation_angle_deg: Rotation angle of camera 2 in degrees
    """
    h, w = img1.shape[:2]



    pixel_cam1 = (w // 2, h // 2)
    u, v = pixel_cam1
    depth_cam1 = depth[v, u]
    
  
    # Mark point in camera 1
    cv2.circle(img1, pixel_cam1, 7, (255, 0, 0), -1)
    cv2.putText(img1, "Camera 1 - Selected Point", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Draw epipolar line WITHOUT depth
    img2_line = draw_epipolar_line_between_cameras(
        pixel_cam1=pixel_cam1,
        K1=K1,
        RT1_w2c=RT1_w2c,
        img_cam2=img2.copy(),
        K2=K2,
        RT2_w2c=RT2_w2c,
        depth_cam1=None,
        color=(0, 255, 0)
    )
    cv2.putText(img2_line, "Camera 2 - Epipolar Line (no depth)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Draw epipolar line WITH depth
    img2_with_point = draw_epipolar_line_between_cameras(
        pixel_cam1=pixel_cam1,
        K1=K1,
        RT1_w2c=RT1_w2c,
        img_cam2=img2.copy(),
        K2=K2,
        RT2_w2c=RT2_w2c,
        depth_cam1=depth_cam1,
        color=(0, 255, 0)
    )
    cv2.putText(img2_with_point, f"Camera 2 - Epipolar Line + Projected Point", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Save individual images
    cv2.imwrite('cam1_point.png', img1)
    cv2.imwrite('cam2_epipolar_line.png', img2_line)
    cv2.imwrite('cam2_with_projection.png', img2_with_point)
    
    # Create matplotlib visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convert BGR to RGB for matplotlib
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_line_rgb = cv2.cvtColor(img2_line, cv2.COLOR_BGR2RGB)
    img2_with_point_rgb = cv2.cvtColor(img2_with_point, cv2.COLOR_BGR2RGB)
    
    # Plot camera 1 with selected point
    axes[0].imshow(img1_rgb)
    axes[0].set_title('Camera 1 - Selected Point', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Plot camera 2 with epipolar line only
    axes[1].imshow(img2_line_rgb)
    axes[1].set_title('Camera 2 - Epipolar Line\n(without depth)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Plot camera 2 with epipolar line and projected point
    axes[2].imshow(img2_with_point_rgb)
    axes[2].set_title('Camera 2 - Epipolar Line + Point\n(with depth)', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('epipolar_geometry_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Images saved!")
    print("  - cam1_point.png: Shows selected point in camera 1")
    print("  - cam2_epipolar_line.png: Shows epipolar line only")
    print("  - cam2_with_projection.png: Shows epipolar line + exact projected point")
    print("  - epipolar_geometry_visualization.png: Combined matplotlib visualization")


def invert_RT(se3: np.ndarray) -> np.ndarray:
        """
        Compute the inverse of a single SE(3) transformation matrix (4x4).

        Args:
            se3 (np.ndarray): A 4x4 SE(3) transformation matrix.

        Returns:
            np.ndarray: The inverse of the input SE(3) matrix.
        """
        assert se3.shape == (4, 4), f"Expected shape (4, 4), got {se3.shape}"

        R = se3[:3, :3]
        T = se3[:3, 3]

        R_inv = R.T
        T_inv = -R_inv @ T

        se3_inv = np.eye(4)
        se3_inv[:3, :3] = R_inv
        se3_inv[:3, 3] = T_inv

        return se3_inv


def compute_visibility_between_cameras(
    image,
    K_ref,
    K_src,
    RTw2c_ref,
    RTw2c_src,
    depth_src,
    num_samples = 512,
    max_depth=8.0,
    decimate=1,
    offset_y=0,
    offset_x=0,
    cuda=False,
    return_numpy=True,
):
    """
    Reference view: view we want supervision for
    Source view: view providing supervision

    Returns:
        visibility_mask: (H, W, num_samples) boolean mask indicating visibility of each sample point
        depth_at_point : (H, W, num_samples)
        points_src_z   : (H, W, num_samples)
        points_ref     : (H, W, num_samples, 3)
    """

    device = torch.device("cuda" if cuda else "cpu")
    dtype = torch.float32

    # ------------------------------------------------------------------
    # Camera parameters
    # ------------------------------------------------------------------
    K_ref = torch.as_tensor(K_ref, dtype=dtype, device=device)
    K_src = torch.as_tensor(K_src, dtype=dtype, device=device)

    ref_to_src = torch.as_tensor(
        RTw2c_src @ invert_RT(RTw2c_ref),
        dtype=dtype,
        device=device,
    )

    # ------------------------------------------------------------------
    # Image grid (possibly decimated)
    # ------------------------------------------------------------------
    H, W = image.shape[:2]

    u, v = torch.meshgrid(
        torch.arange(W, device=device, dtype=dtype),
        torch.arange(H, device=device, dtype=dtype),
        indexing="xy",
    )

    if decimate > 1:
        u = u[offset_y::decimate, offset_x::decimate]
        v = v[offset_y::decimate, offset_x::decimate]

    H_d, W_d = u.shape

    ones_hw = torch.ones((H_d, W_d, 1), device=device)
    ones_hws = torch.ones((H_d, W_d, num_samples, 1), device=device)

    # Ray directions in reference camera
    uv1 = torch.cat((u[..., None], v[..., None], ones_hw), dim=-1)  # (H, W, 3)

    K_ref_inv = torch.linalg.inv(K_ref)
    rays = uv1 @ K_ref_inv.T
    rays = rays / torch.linalg.norm(rays, dim=-1, keepdim=True)

    # Sample points along rays
    scales = torch.linspace(0.0, max_depth, num_samples, device=device) # (N)
    points_ref = rays[..., None, :] * scales[None, None, :, None]  # (H,W,N,3)

    # Transform points to source camera
    # 
    points_ref_h = torch.cat((points_ref, ones_hws), dim=-1)
    points_src = (points_ref_h @ ref_to_src.T)[..., :3]

    # ------------------------------------------------------------------
    # Project into source image
    # ------------------------------------------------------------------
    proj = points_src @ K_src.T
    points_src_z = proj[..., 2]
    proj_xy = proj[..., :2] / proj[..., 2:3]
    pixel_coords = proj_xy.clone()




    # Normalize to [-1, 1] for grid_sample
    proj_xy[..., 0] = proj_xy[..., 0] / W * 2 - 1
    proj_xy[..., 1] = proj_xy[..., 1] / H * 2 - 1

    # ------------------------------------------------------------------
    # Sample source depth
    # ------------------------------------------------------------------
    depth_src = torch.as_tensor(depth_src, dtype=dtype, device=device)
    depth_src = depth_src.unsqueeze(0).repeat(num_samples, 1, 1, 1)

    grid = proj_xy.permute(2, 0, 1, 3)  # (N,H,W,2)

    depth_at_point = F.grid_sample(
        depth_src,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).squeeze(1)

    depth_at_point = depth_at_point.permute(1, 2, 0)

    # Validity mask (inside image)
    x = pixel_coords[..., 0]
    y = pixel_coords[..., 1]
    in_bounds = (x >= 0) & (x < W) & (y >= 0) & (y < H)
    depth_at_point[~in_bounds] = 0.00
    depth_at_point[depth_at_point <= 0] = 0.00

    visibility = (depth_at_point > points_src_z) & (depth_at_point > 0.00)& (points_src_z > 0)


    if return_numpy:
        return (visibility.cpu().numpy(),depth_at_point.cpu().numpy(),points_src_z.cpu().numpy(),points_ref.cpu().numpy(), pixel_coords.cpu().numpy())

    return visibility, depth_at_point, points_src_z, points_ref



def classify_visibility_events(depth_gt, depth_proj, visibility):
    """
    check visibility events.

    depth_gt, depth_proj: (H,W,N)
    visibility: boolean, (H,W,N)

    Returns:
      events_full (H,W,N)
        0  : no event
        1  : disocclusion
       -1  : occlusion
       -2  : invalid
    """

    vis_int = visibility.astype(np.int8)

    valid_mask = np.isfinite(depth_gt)

    # visibility transitions (assigned to index k)
    events = np.diff(vis_int, axis=-1)          # (H,W,N-1)
    events = np.pad(events, ((0, 0), (0, 0), (1, 0)), mode="constant")

    eps = 0.02 * np.maximum(depth_gt, depth_proj)

    # depth agreement at sample k
    depth_agree = np.abs(depth_gt - depth_proj) <= eps  # (H,W,N)

    events_full = np.zeros_like(vis_int)

    # only classify real transitions 
    transition = events != 0

    events_full[transition & depth_agree & valid_mask]  = 1   # disocclusion/ intersection
    events_full[transition & ~depth_agree & valid_mask] = -1  # occlusion
    events_full[valid_mask == False] = -2        # mark invalid

    # breakpoint()
    return events_full


@dataclass
class RayEventSegment:
    """
    Represents a segment along a camera ray between two visibility events.
    
    Attributes:
        x: Pixel x-coordinate (width)
        y: Pixel y-coordinate (height)
        start: Starting sample index along the ray
        end: Ending sample index along the ray
        start_dist: Distance from camera to start of segment
        end_dist: Distance from camera to end of segment
        eventType: Type of event pair (1=II, 2=IO, 3=OI, 4=OO)
    """
    x: int
    y: int
    start: int
    end: int
    start_dist: float
    end_dist: float
    eventType: int # {1: II, 2: I0, 3: OI, 4: OO}


def count_valid_segments_from_indices(
    h,
    w, 
    event_ray,
    validity_mask,
    min_seg_len=5,
):
    H, W, ray_len,  = validity_mask.shape
    is_event = (event_ray == 1) | (event_ray == -1) 

    ray = event_ray[h, w, :]
    
    if np.any(ray == -2):
        print("hitting here")
        return 0

    valid_idxs = np.where(is_event[h, w, :])[0] 
    valid_idxs = list(valid_idxs)
    print(valid_idxs)


    if len(valid_idxs) == 0:
        return 0

    assert all(
        valid_idxs[i] < valid_idxs[i + 1] for i in range(len(valid_idxs) - 1)
    ), "valid_idxs must be strictly increasing"

    segments = []
    n = len(valid_idxs)

    # -----------------------------------
    # If odd count → reserve last for tail
    # -----------------------------------
    has_tail = (n % 2 == 1)
    last_tail_idx = valid_idxs[-1] if has_tail else None

    # -----------------------------------
    # Pair interior indices only
    # -----------------------------------
    end = n - 1 if has_tail else n
    for i in range(0, end, 2):
        s = valid_idxs[i]
        e = valid_idxs[i + 1]
        segments.append((s, e))

    # -----------------------------------
    # Add tail segment
    # -----------------------------------
    if has_tail:
        assert last_tail_idx < ray_len, "Invalid index beyond ray length"
        if validity_mask[h, w, -1]:
            segments.append((last_tail_idx, ray_len - 1))

    # -----------------------------------
    # Validate segments
    # -----------------------------------
    count = 0
    for s, e in segments:
        if ((e - s) + 1) >= min_seg_len:
            count += 1
    

    return count


def find_consecutive_ones(v):
        """
        Find start and end indices of consecutive 1s in a 1D binary array.

        Args:
            v: (N,) array of {0,1} or bool

        Returns:
            segments: list of (start_idx, end_idx)
        """
        v = np.asarray(v).astype(np.uint8)

        starts = np.where(
            (v == 1) &
            (np.concatenate(([0], v[:-1])) == 0)
        )[0]

        ends = np.where(
            (v == 1) &
            (np.concatenate((v[1:], [0])) == 0)
        )[0]

        return list(zip(starts, ends))



def generate_supervision_for_single_pixel(
    pixel_loc,
    depth_gt,
    depth_proj,
    visibility,
    min_segments=1,
    min_segment_length=6,
    max_depth=16,
):
    """
    Generate ray event supervision for a single pixel.

    Args:
        pixel_loc: (h, w) pixel coordinates
        depth_gt: (H, W, N) ground-truth depth along rays
        depth_proj: (H, W, N) projected depth along rays
        visibility: (H, W, N) visibility mask
        min_segments: minimum number of valid segments required (unused here but kept)
        min_segment_length: minimum length of a valid segment
        max_depth: maximum ray depth

    Returns:
        List[RayEventSegment]
    """

    h, w = pixel_loc
    _, _, N = depth_gt.shape

    visibility_mask = visibility.astype(np.uint8)

    # Depth consistency tolerance
    eps = 0.01 + 0.02 * np.maximum(depth_gt, depth_proj)

    # Distances sampled along the ray
    dist_along_ray = np.linspace(0.0, max_depth, N)

    # Event encoding
    symbol_map = {True: "I", False: "O"}   # Intersection / Occlusion
    event_id_map = {"II": 1, "IO": 2, "OI": 3, "OO": 4}

    # Visibility along this pixel's ray
    vis_ray = visibility_mask[h, w, :]

    # Find visible segments
    segments = find_consecutive_ones(vis_ray)

    pixel_segments = []

    for start_idx, end_idx in segments:
        if (end_idx - start_idx + 1) < min_segment_length:
            continue

        # Depth consistency at segment boundaries
        start_match = (
            abs(depth_gt[h, w, start_idx] - depth_proj[h, w, start_idx])
            <= eps[h, w, start_idx]
        )
        end_match = (
            abs(depth_gt[h, w, end_idx] - depth_proj[h, w, end_idx])
            <= eps[h, w, end_idx]
        )

        event_type = event_id_map[
            symbol_map[start_match] + symbol_map[end_match]
        ]

        pixel_segments.append(
            RayEventSegment(
                x=h,
                y=w,
                start=int(start_idx),
                end=int(end_idx),
                start_dist=float(dist_along_ray[start_idx]),
                end_dist=float(dist_along_ray[end_idx]),
                eventType=event_type,
            )
        )

    return pixel_segments


def generate_supervision_data(
    depth_gt: np.ndarray,
    depth_proj: np.ndarray,
    visibility: np.ndarray,
    min_segment_length: int = 5,
    min_num_segments: int = 1,
    max_depth: float = 16.0,
):
    """
    Generate ray-event supervision over the entire image using visibility segments.
    """

    H, W, N = depth_gt.shape

    visibility_mask = visibility.astype(np.uint8)

    # Depth consistency tolerance
    eps = 0.01 + 0.02 * np.maximum(depth_gt, depth_proj)

    # Distances along ray
    dist_along_ray = np.linspace(0.0, max_depth, N)

    # Event encodings
    symbol_map = {True: "I", False: "O"}
    event_id_map = {"II": 1, "IO": 2, "OI": 3, "OO": 4}

    # Outputs
    segments_per_pixel = [[[] for _ in range(W)] for _ in range(H)]
    valid_pixel_mask = np.zeros((H, W), dtype=bool)

    for h in range(H):
        for w in range(W):

            vis_ray = visibility_mask[h, w, :]

            # Find visible segments
            segments = find_consecutive_ones(vis_ray)
            if len(segments) == 0:
                continue

            pixel_segments = []

            for start_idx, end_idx in segments:
                segment_len = end_idx - start_idx + 1
                if segment_len < min_segment_length:
                    continue

                # Depth agreement at boundaries
                start_match = (
                    abs(depth_gt[h, w, start_idx] - depth_proj[h, w, start_idx])
                    <= eps[h, w, start_idx]
                )
                end_match = (
                    abs(depth_gt[h, w, end_idx] - depth_proj[h, w, end_idx])
                    <= eps[h, w, end_idx]
                )

                event_type = event_id_map[
                    symbol_map[start_match] + symbol_map[end_match]
                ]

                pixel_segments.append(
                    RayEventSegment(
                        x=h,
                        y=w,
                        start=int(start_idx),
                        end=int(end_idx),
                        start_dist=float(dist_along_ray[start_idx]),
                        end_dist=float(dist_along_ray[end_idx]),
                        eventType=event_type,
                    )
                )

            if len(pixel_segments) >= min_num_segments:
                valid_pixel_mask[h, w] = True
                segments_per_pixel[h][w].extend(pixel_segments)

        

    return segments_per_pixel, valid_pixel_mask



def total_segment_events_loss(
    y_pred,             # (N,)
    gt_segment_data,    # (S, 3): [start_z, end_z, event_type]
    locations,          # (N,)
    reduction="none",
):
    S = gt_segment_data.shape[0]
    N = y_pred.shape[0]

    y_pred_exp = y_pred.unsqueeze(0).expand(S, N)          # (S, N)
    locations_exp = locations.unsqueeze(0).expand(S, N)   # (S, N)

    gt_distances = gt_segment_data[:, :2]                 # (S, 2)
    segment_event_types = gt_segment_data[:, 2].long()   # (S,)

    # Validity mask: inside segment
    validity_mask = (
        (locations_exp > gt_distances[:, 0].unsqueeze(1)) &
        (locations_exp <= gt_distances[:, 1].unsqueeze(1))
    ).float()  # (S, N)

    # Signed distances

    below = gt_distances[:, 0].unsqueeze(1) -  locations_exp 
    above =  gt_distances[:, 1].unsqueeze(1) - locations_exp

    sCloser = torch.abs(below) < torch.abs(above)  # (S, N)


    # -----------------------------
    # Compute losses per event type
    # -----------------------------
    i1_loss = torch.abs(y_pred_exp - below)
    i2_loss = torch.abs(y_pred_exp - above)

    # II
    loss_ii = i1_loss * (sCloser == True) + i2_loss * (sCloser == False)

    # OO
    mid = (below + above) / 2
    halfWidth = (above - below) / 2
    diff = torch.abs(y_pred_exp - mid)
    loss_oo = F.relu(halfWidth - diff)


    # IO
    o2_loss = torch.minimum(F.relu(above - y_pred_exp), i1_loss)
    loss_io = i1_loss * (sCloser == True) + o2_loss * (sCloser == False)

    # OI
    o1_loss = torch.minimum(F.relu(y_pred_exp - below), i2_loss)
    loss_oi = o1_loss * (sCloser == True) + i2_loss * (sCloser == False)


    # -----------------------------
    # Apply validity mask (out-of-place!)
    # -----------------------------
    loss_ii = loss_ii * validity_mask
    loss_oo = loss_oo * validity_mask
    loss_io = loss_io * validity_mask
    loss_oi = loss_oi * validity_mask

    # Select correct loss per segment
    mask_ii = (segment_event_types == 1).unsqueeze(1).expand(-1, N)
    mask_io = (segment_event_types == 2).unsqueeze(1).expand(-1, N)
    mask_oi = (segment_event_types == 3).unsqueeze(1).expand(-1, N)
    mask_oo = (segment_event_types == 4).unsqueeze(1).expand(-1, N)

    total_loss = (
        loss_ii * mask_ii +
        loss_io * mask_io +
        loss_oi * mask_oi +
        loss_oo * mask_oo
    )

    z = locations_exp.cpu().squeeze().numpy()
    a = above.cpu().squeeze().numpy()
    b = below.cpu().squeeze().numpy()

    # Reduction
    if reduction == "mean":
        return total_loss.mean()
    elif reduction == "segment":
        return total_loss.mean(dim=1)  # (S,)
    elif reduction == "none":
        return total_loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")





def group_segments_by_pixel_with_min_count(
    segments: List[RayEventSegment],
    min_segments: int = 2,
) -> Dict[str, List[RayEventSegment]]:
    """
    Groups ray event segments by pixel using key 'x_y'
    and keeps only pixels with at least `min_segments` segments.
    """

    pixel_to_segments = defaultdict(list)

    # Group by pixel using string key
    for seg in segments:
        key = f"{seg.x}_{seg.y}"
        pixel_to_segments[key].append(seg)

    # Filter pixels with >= min_segments
    filtered = {
        key: segs
        for key, segs in pixel_to_segments.items()
        if len(segs) >= min_segments
    }

    return filtered


def build_segments_and_targets(
    segments: List['RayEventSegment'],
):
    """
    Builds gt_segment_data for one ray.

    Returns:
        gt_segment_data: (S, 3)
            [:, 0] = start_dist
            [:, 1] = end_dist
            [:, 2] = eventType (float)
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    S = len(segments)

    # (S, 3): start, end, event type
    gt_segment_data = torch.zeros(S, 3, device=device)

    for s, seg in enumerate(segments):
        gt_segment_data[s, 0] = seg.start_dist
        gt_segment_data[s, 1] = seg.end_dist
        gt_segment_data[s, 2] = float(seg.eventType)

    return gt_segment_data



def segments_list_to_tensor(sampled_segments, N_rays_points):
    """
    Convert list[RayEventSegment] to vectorized tensors.

    Returns:
        segments:      (B, N, 2)
        target_types:  (B, N)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    B = len(sampled_segments)
    N = N_rays_points

    segments = torch.zeros((B, N, 2), dtype=torch.long, device=device)
    target_types = torch.zeros((B,), dtype=torch.long, device=device)

    for b, seg in enumerate(sampled_segments):
        segments[b, :, 0] = seg.start
        segments[b, :, 1] = seg.end
        target_types[b] = seg.eventType

    return segments, target_types








def compute_penalty_field(segment, z_vals, y_vals):
    """
    Returns: (num_y, num_z) penalty grid
    """
    # breakpoint()
    penalty = torch.zeros((len(y_vals), len(z_vals)), device=z_vals.device)


    for j, y in enumerate(y_vals):
        # DRDF prediction along the ray: constant y
        y_pred = torch.full_like(z_vals, y)
        # breakpoint()

        loss = ray_cost_function(
            y_pred=y_pred,
            segment=segment,
            z_vals = z_vals,
            # num_points=512,
            reduction="none"   
        )

        penalty[j] = loss

    return penalty




def generate_rays_with_samples(
    depth,
    K,
    RT_w2c,
    n_samples=512,
    max_depth=8.0
):
    """
    Generate rays and sample points along each ray in world coordinates.

    Returns:
        ray_origins: (H, W, 3)
        ray_directions: (H, W, 3)
        sampled_points: (H, W, n_samples, 3)
    """

    H, W = depth.shape

    u = np.arange(W)
    v = np.arange(H)
    uu, vv = np.meshgrid(u, v)

    ones = np.ones_like(uu)
    pix_h = np.stack([uu, vv, ones], axis=-1)  # (H, W, 3)

    #  Backproject to camera rays
    K_inv = np.linalg.inv(K)

    pix_h = rearrange(pix_h, 'h w  c -> (h w ) c')  # (H*W*N, 3)

    cam_dirs = (K_inv @ pix_h.T).T  # (H*W, 3)
    cam_dirs = cam_dirs / np.linalg.norm(cam_dirs, axis=-1, keepdims=True)
    cam_dirs = rearrange(cam_dirs, '(h w) c -> h w c', h=H, w=W)  # (H, W, 3)

    #  Camera → world transform
    RT_c2w = invert_RT(RT_w2c)
    R_c2w = RT_c2w[:3, :3]
    t_c2w = RT_c2w[:3, 3]

    # Ray origins (camera center)
    ray_origins = np.broadcast_to(
        t_c2w.reshape(1, 1, 3),
        (H, W, 3)
    )

    # Ray directions in world frame
    ray_directions = cam_dirs @ R_c2w

    ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=-1, keepdims=True)
   
    # Sample points along rays
    t_vals = np.linspace(0.1, max_depth, n_samples)

    sampled_points = (
        ray_origins[..., None, :] +
        ray_directions[..., None, :] * t_vals[None, None, :, None]
    )  # (H, W, n_samples, 3)

    return ray_origins, ray_directions, sampled_points


# import numpy as np
# from scipy.ndimage import map_coordinates

def compute_visibility_mask(
    sampled_points,  # (H, W, N, 3) world coordinates
    depth,           # (H, W) depth map of the reference camera
    K,               # (3, 3) intrinsic of reference camera
    RT_w2c,          # (4, 4) world to camera pose
    depth_thresh=0.01
):
    """
    Project sampled world points into reference camera and check depth consistency.

    Uses bilinear sampling to reduce jagged edges in visibility mask.

    Returns:
        visibility_mask: (H, W, N) boolean
        visibility_max: (H, W) boolean
    """
    H, W, N, _ = sampled_points.shape

    # Convert sampled points to homogeneous coordinates
    ones = np.ones((H, W, N, 1), dtype=sampled_points.dtype)
    pts_world_h = np.concatenate([sampled_points, ones], axis=-1)  # (H,W,N,4)

    pts_world_h_flat = rearrange(pts_world_h, 'h w n c -> (h w n) c')  # (H*W*N, 4)

    # World → camera (homogeneous)
    pts_cam_h = (RT_w2c @ pts_world_h_flat.T).T  # (H,W,N,4)

    pts_cam_h = rearrange(pts_cam_h, '(h w n) c -> h w n c', h=H, w=W, n=N)  # (H, W, N, 4)
    pts_cam = pts_cam_h[..., :3]
    z = pts_cam[..., 2]  # camera depth

    # Flatten H, W, N into a single dimension
    pts_cam_flat = rearrange(pts_cam, 'h w n c -> (h w n) c')  # (H*W*N, 3)

    # Apply intrinsic matrix
    proj_flat = (K @ pts_cam_flat.T).T  # (H*W*N, 3)
    proj = rearrange(proj_flat, '(h w n) c -> h w n c', h=H, w=W, n=N)  # (H, W, N, 3)

    # Camera → image plane
    u = proj[..., 0] / proj[..., 2]
    v = proj[..., 1] / proj[..., 2]

    proj_xy = np.stack([u, v], axis=-1)  # (H, W, N, 2)
    proj_xy[..., 0] = proj_xy[..., 0] / W * 2 - 1  # x: [0, W] → [-1, 1]
    proj_xy[..., 1] = proj_xy[..., 1] / H * 2 - 1  # y: [0, H] → [-1, 1]

    # Valid projection mask
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)

    # Bilinear sampling of reference depth
    coords = np.stack([v, u], axis=0)  # (2, H, W, N)
    coords_flat = coords.reshape(2, -1)  # (2, H*W*N)
    depth_sampled_flat = map_coordinates(depth, coords_flat, order=1, mode='nearest')
    ref_depth_sampled = depth_sampled_flat.reshape(H, W, N)

    # Depth consistency check
    visibility = (valid &(ref_depth_sampled > (z + depth_thresh)))

    return visibility


# # def compute_visibility_mask(
# #     sampled_points,      # (H, W, N, 3) world coords
# #     depth,           # (H, W)
# #     K,               # (3, 3)
# #     RT_w2c,          # (4, 4)
# #     depth_thresh=1e-5
# # ):
# #     """
# #     Project sampled world points into reference camera and check depth consistency.

# #     Returns:
# #         visibility_mask: (H, W, N) boolean
# #         visibility_max: (H, W) boolean
# #     """

# #     H, W, N, _ = sampled_points.shape

# #     # Convert sampled points to homogeneous coords
# #     ones = np.ones((H, W, N, 1), dtype=sampled_points.dtype)
# #     pts_world_h = np.concatenate([sampled_points, ones], axis=-1)  # (H,W,N,4)

# #     # World → camera (homogeneous)
# #     pts_cam_h = pts_world_h @ RT_w2c.T   # (H,W,N,4)
# #     pts_cam = pts_cam_h[..., :3]

# #     z = pts_cam[..., 2]  # camera depth

# #     #  Camera → image plane
# #     proj = pts_cam @ K.T  # (H,W,N,3)

# #     u = proj[..., 0] / proj[..., 2]
# #     v = proj[..., 1] / proj[..., 2]

# #     # u = np.round(u).astype(np.int64)
# #     # v = np.round(v).astype(np.int64)

# #     #  Valid projection mask
# #     valid = (
# #         (u >= 0) & (u < W) &
# #         (v >= 0) & (v < H) &
# #         (z > 0)
# #     )

    
# #     # bilinear sampling
# #     coords = np.stack([v, u], axis=0)  # shape (2, H*W*N)
# #     ref_depth_sampled = map_coordinates(depth, coords, order=1, mode='nearest').reshape(H, W, N)

# #     # Sample reference depth
# #     ref_depth = np.zeros_like(z)
# #     ref_depth[valid] = depth[v[valid], u[valid]]

# #     #  Depth consistency check
# #     visibility = (valid &(ref_depth > (z + depth_thresh)))

# #     return visibility




