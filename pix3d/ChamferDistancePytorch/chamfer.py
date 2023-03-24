import torch, chamfer3D.dist_chamfer_3D, fscore
from plyfile import PlyData, PlyElement
chamLoss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
#https://github.com/ThibaultGROUEIX/ChamferDistancePytorch/tree/a1487a475912b598b1eaf96fafbb0e2591eee86b
if __name__ == "__main__":
    with open('../gt_ply/sofa.ply', 'rb') as f:
        plydata = PlyData.read(f)

    with open('../3dr2n2_ply/sofa.ply', 'rb') as f:
        plydata2 = PlyData.read(f)

    # with open('model.ply', 'rb') as f:
    #     plydata = PlyData.read(f)

    # with open('model2.ply', 'rb') as f:
    #     plydata2 = PlyData.read(f)
    
    print(plydata)
    points = []
    for item in plydata.elements[0].data:
      points.append(item)
    points = torch.FloatTensor(points)
    n, _ = points.shape
    points = torch.unsqueeze(points, 0)
    print(points.shape)

    points2 = []
    for item in plydata2.elements[0].data:
      points2.append(item)
    points2 = torch.FloatTensor(points2)
    n, _ = points2.shape
    points2 = torch.unsqueeze(points2, 0)
    print(points2.shape)

    points1 = points.cuda()
    points2 = points2.cuda()
    dist1, dist2, idx1, idx2 = chamLoss(points1, points2)
    f_score, precision, recall = fscore.fscore(dist1, dist2)

    # print("here")
    print("dist1:", dist1, "\ndist2:", dist2, "\nidx1:", idx1, "\nidx2:",idx2)
    print("chamfer distance", torch.sum(dist1) + torch.sum(dist2))
    print("f_score:", f_score, "\nprecision:", precision, "\nrecall:",recall)