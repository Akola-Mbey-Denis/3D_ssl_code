import numpy as np
import time
from PIL import Image, ImageDraw
import torch
import os
import pickle 
import matplotlib.pyplot as plt
import pdb

def homogenize(x, dim):
    assert dim in [0,1]
    if dim == 1:
        return np.hstack([x, np.ones((x.shape[0], 1))])
    return np.vstack([x, np.ones((1, x.shape[1]))])

def dehomogenize(x, dim):
    assert dim in [0,1]
    if dim == 1:
        return x[:,:-1] / x[:,-1]
    return x[:-1,:] / x[-1,:]

def allPointsFromImKD(I, K, D):
    # take an image I, intrinsics K, and depth D
    # return: (1) a 3x(HW) set of points; (2) (HW)x3 rgb
    H, W = I.shape[0], I.shape[1]
    
    pointsX, pointsY = np.meshgrid(np.arange(W).astype(float), np.arange(H).astype(float))
    uvAll = np.vstack([pointsX.reshape(1,-1), pointsY.reshape(1,-1)])
    xyzAll = np.linalg.inv(K) @ homogenize(uvAll, 0)
    xyzAll = D.reshape(1, H*W) * xyzAll
    colorsJoin = I.reshape(HRef*WRef, 3)

    k = D.ravel() > 0

    return xyzAll[:,k], colorsJoin[k,:]


def getVisibility(I, KRef, KSrc, PW2CRef, PW2CSrc, DSrc, maxZ, numSamples, 
           decimate=1, offsetY=0, offsetX=0, cuda=False, returnNumpy=True):
    """
    Definition: 
    - reference view is the view for which we want the supervision
    - source view is the view that is the source of the supervision

    Given:
        - image I  (just used for size for the output)
        - KRef: reference view intrinsics
        - KSrc: source view intrinsics
        - PW2CRef: 4x4 world to camera extrinsics pose for reference view
        - PW2CSrc: 4x4 world to camera extrinsics pose for source view
        - DSrc: depth for the source image 
        - maxZ: maximum depth
        - numSamples: number of samples from 0 to maxZ
    
    Optional
        - decimate: decimate the image by this factor (1 = no decimation)
        - offsetY, offsetX: shift before decimation
        - cuda: whether to use a gpu
        - returnNumpy: if true, send back to cpu and put in numpy; if false, return cuda on gpu

    If I is H x W, this returns:
    - depthAtPoint: H x W x numSamples giving the source view depth at that point
    - pointsSrcZ: H x W x numSamples giving the ray's depth to the source camera at that point
    - points: H x W x numSamples x 3 giving the 3D location in the reference view of that point

    Note that:
    (a) depthAtPoint is *not* on a Euclidean grid and will not look correct. It needs to be plotted
    using the points variable
    (b) decimate will make the return H//decimate and W//decimate
    """

    device = "cpu" if not cuda else "cuda"

    KRef = torch.from_numpy(KRef).to(torch.float32).to(device=device)
    KSrc = torch.from_numpy(KSrc).to(torch.float32).to(device=device)

    # right multiplying gives ref to src
    Ref2Src = torch.from_numpy(PW2CSrc @ np.linalg.inv(PW2CRef))
    Ref2Src = Ref2Src.to(torch.float32).to(device)

    # H, W are the actual image size. Note that U, V might be smaller if decimate
    # mode is on
    H, W = I.shape[0], I.shape[1]

    U, V = torch.meshgrid(
            torch.arange(W, dtype=torch.float32, device=device), 
            torch.arange(H, dtype=torch.float32, device=device), 
            indexing='xy')

    if decimate > 1:
        U = U[offsetY::decimate, offsetX::decimate ]
        V = V[offsetY::decimate, offsetX::decimate ]

    HWOnes = torch.ones(U.shape[0], U.shape[1], device=device)[:,:,None]
    HWOnesSamp = torch.ones(U.shape[0], U.shape[1], numSamples, 1, device=device)

    # generate the scales
    scale = torch.linspace(0, maxZ, numSamples, device=device)

    # H x W x 3 UV map
    UV = torch.dstack([U[:,:,None], V[:,:,None], HWOnes])

    # H x W x rays
    KRefInv = torch.linalg.inv(KRef)
    rays = (UV @ KRefInv.T)
    rays = rays / torch.linalg.vector_norm(rays, dim=2, keepdim=True)

    # create points from scaled rays
    points = rays[:,:,None,:] * scale[None, None, :, None]

    pointsSrc = (torch.cat([points, HWOnesSamp], axis=3) @ Ref2Src.T)[:,:,:,:3]
    pointsSrcProject = (pointsSrc @ KSrc.T)
    pointsSrcZ = pointsSrcProject[:,:,:,2] # depth
    pointsSrcProject = pointsSrcProject[:,:,:,:2] / pointsSrcProject[:,:,:,[2]]

    # normalize to [-1, 1]
    pointsSrcProject[:,:,:,0] = (pointsSrcProject[:,:,:,0] / W) * 2 - 1
    pointsSrcProject[:,:,:,1] = (pointsSrcProject[:,:,:,1] / H) * 2 - 1 

    DSrc = torch.from_numpy(DSrc).to(torch.float32).to(device)
    DSrc = DSrc.expand(numSamples, 1, DSrc.shape[0], DSrc.shape[1])


    depthAtPoint = torch.nn.functional.grid_sample(
                    DSrc,
                    pointsSrcProject.permute(2, 0, 1, 3),
                    mode='bilinear', padding_mode='zeros', 
                    align_corners=False).squeeze(1)
    depthAtPoint = depthAtPoint.permute(1, 2, 0)

    DSrcOnes = torch.ones(DSrc.shape, dtype=torch.float32, device=device)
    DSrcOnes = DSrcOnes.expand(*DSrc.shape)

    inImage = torch.nn.functional.grid_sample(
                DSrcOnes, 
                pointsSrcProject.permute(2, 0, 1, 3),
                mode='bilinear', padding_mode='zeros',
                align_corners=False).squeeze(1)
    inImage = inImage.permute(1, 2, 0)

    depthAtPoint[inImage < 1] = torch.nan

    if returnNumpy:
        return depthAtPoint.cpu().numpy(), pointsSrcZ.cpu().numpy(), points.cpu().numpy()
    else:
        return depthAtPoint, pointsSrcZ, points

if __name__ == "__main__":
    X = pickle.load(open("/scratch/projects/fouheylab/dma9300/occ_project/occ/matterport3d_data.pkl","rb"))

    ref = X['target_view']
    sources = X['source_views']

    IRef = ref['image']
    DRef = ref['depth']
    KRef = ref['intrinsics']
    RtRefW2C = ref['extrinsics']
    HRef, WRef = IRef.shape[0], IRef.shape[1]

    u, v = 558 + 10, 527 + 30
    u, v = 558 + 10, 527 - 150
    u, v = 558 - 270, 527 + 100
#    u, v = 558 + 200, 527 + 100

    # draw the point
    IRefIm = Image.fromarray(IRef)
    IRefD = ImageDraw.Draw(IRefIm)
    IRefD.ellipse((u-5, v-5, u+5, v+5), fill=(255,0,0))
    IRefIm.save("visSrc.png")

    r = np.linalg.inv(KRef) @ np.array([[u, v, 1]]).T
    r = r / np.linalg.norm(r)
    rayScale = np.linspace(0,8,64)
    pointsRef = rayScale * r

    xyzAllRef, colorsRef = allPointsFromImKD(IRef, KRef, DRef)

    allVis = []

    for sourceI, source in enumerate(sources):
        ISrc = source['image']
        DSrc = source['depth']
        KSrc = source['intrinsics']
        RtSrcW2C = source['extrinsics']
        HSrc, WSrc = ISrc.shape[0], ISrc.shape[1]


        plt.imsave("d.png", DSrc, vmin=0, vmax=8, cmap='plasma')

        Ref2SrcC2C = RtSrcW2C @ np.linalg.inv(RtRefW2C)

        pointsSrc = dehomogenize(Ref2SrcC2C @ homogenize(pointsRef, 0), 0)

        pointsSrcProject = KSrc @ pointsSrc
        pointsSrcZ = pointsSrcProject[2,:] # get depth
        pointsSrcProject = dehomogenize(pointsSrcProject, 0) # get pixels

        # print(pointsSrcProject[0,:])

        visibility = np.zeros((pointsSrcZ.size,), dtype=bool)
        actualDepth = np.zeros((pointsSrcZ.size,), dtype=float)

        ISrcDIm = Image.open("d.png")
        ISrcIm = Image.fromarray(ISrc)
        ISrcDD = ImageDraw.Draw(ISrcDIm)
        ISrcD = ImageDraw.Draw(ISrcIm)

        for i in range(pointsSrcProject.shape[1]):
            p = pointsSrcProject[:,i]
            pInt = int(p[0]), int(p[1])
            z = pointsSrcZ[i]
            visible = False
            inFront = z > 0
            if (pInt[0] > 0) and (pInt[1] > 0) and (pInt[0] <= WSrc-1) and (pInt[1] <= HSrc-1):
                visible = (z < DSrc[pInt[1], pInt[0]])
                actualDepth[i] = DSrc[pInt[1], pInt[0]]

            visibility[i] = visible

            col = (0,0,0)
            if inFront:
                col = (255,0,0) if visible else (0,0,255)

            ISrcDD.point((p[0], p[1]), fill=col)
            ISrcD.point((p[0], p[1]), fill=col)

        ISrcIm.save("visTarget.png")
        ISrcDIm.save("visTargetD.png")

        plt.plot(actualDepth)
        plt.plot(pointsSrcZ)
        plt.savefig("lines.pdf")
        plt.close()

        xyzAllSrc, colorsSrc = allPointsFromImKD(ISrc, KSrc, DSrc)

        xyzAllRefInSrc = dehomogenize(Ref2SrcC2C @ homogenize(xyzAllRef, 0), 0)


        cm = plt.get_cmap('viridis')
        plt.scatter(xyzAllSrc[0][::4], xyzAllSrc[1][::4], xyzAllSrc[2][::4], colorsSrc[::4]/255)
        plt.scatter(pointsSrc[0], pointsSrc[1], pointsSrc[2], c=rayScale)
        plt.gca().yaxis.set_inverted(True)
        plt.savefig("vis.png")
        plt.close()

        print(actualDepth)
        tic = time.time()
        zRecorded, zRay, points3D = getVisibility(IRef, KRef, KSrc, RtRefW2C, RtSrcW2C, DSrc, 16.0, 512, decimate=2)
        toc = time.time()
        print("done in %.2fs" % (toc-tic))

        vis = (zRecorded > zRay) & (~np.isnan(zRecorded)) & (zRay > 0) 
        allVis.append(vis)

        plt.imsave("slice_%d.png" % sourceI, vis[256,:,:])

    anyVisible = allVis[0]
    for i in range(1, len(allVis)):
        anyVisible = anyVisible | allVis[i]

    # get a slice in the center of the image
    locationSlice = points3D[256] 
    anyVisibleSlice = anyVisible[256] 

    locationSliceX = locationSlice[:,:,0]
    locationSliceZ = locationSlice[:,:,2]

    # scatter the x z coordinates
    plt.scatter(locationSliceX[anyVisibleSlice], locationSliceZ[anyVisibleSlice], s=4)
    plt.savefig("vis.png")

    pdb.set_trace()