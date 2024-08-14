# https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py

import torch as t


def square_distance(points1, points2):
    batch_size, n, channels = points1.shape
    _, m, _ = points2.shape
    dist = -2 * t.matmul(points1, points2.permute(0, 2, 1))
    dist += t.sum(points1 ** 2, -1).view(batch_size, n, 1)
    dist += t.sum(points2 ** 2, -1).view(batch_size, 1, m)
    return dist


def ball_query(point_cloud, centroids, radius, samples_count):
    device = point_cloud.device
    batch_size, points_count, channels = point_cloud.shape
    _, centroids_count, _ = centroids.shape

    sampled_cloud = point_cloud.view(batch_size, 1, points_count, channels).repeat(1, centroids_count, 1, 1)
    #sampled_cloud_indexes = t.arange(points_count, dtype=t.long, device=point_cloud.device).view(1, 1, points_count).repeat((batch_size, centroids_count, 1))

    MAX = t.Tensor([float('inf')] * channels).to(device,  dtype=t.float)

    if channels > 3:
        distances = square_distance(centroids, point_cloud[:, :, : 3])
    else:
        distances = square_distance(centroids, point_cloud)
    #sampled_cloud_indexes[(distances > radius ** 2)] = points_count
    sampled_cloud[(distances > radius ** 2)] = MAX

    sort_indexes = sampled_cloud[:, :, :, 0].sort(dim=-1, stable=True)[1].flatten()
    batch_indexes = t.arange(batch_size, device=device).repeat_interleave(centroids_count * points_count)
    centroids_indexes = t.arange(centroids_count, device=device).repeat(batch_size).repeat_interleave(points_count)

    sampled_cloud = sampled_cloud[batch_indexes, centroids_indexes, sort_indexes].view(batch_size, centroids_count,
                                                                                       points_count, channels)[:, :, :samples_count]
    query = sampled_cloud == MAX
    first = sampled_cloud[:, :, 0].view(batch_size, centroids_count, 1, channels).repeat(1, 1, samples_count, 1)
    sampled_cloud[query] = first[query]

    """sampled_cloud_indexes, indexes = sampled_cloud_indexes.sort(dim=-1)
    #print(sampled_cloud[indexes])
    sampled_cloud_indexes = sampled_cloud_indexes[:, :, :samples_count]
    query = sampled_cloud_indexes == points_count
    first = sampled_cloud_indexes[:, :, 0].view(batch_size, centroids_count, 1).repeat(1, 1, samples_count)

    sampled_cloud_indexes[query] = first[query]"""

    return sampled_cloud #, sampled_cloud_indexes


def farthest_point_sampling(point_cloud, centroids_count):
    device = point_cloud.device
    batch_size, points, channels = point_cloud.shape
    if channels > 3:
        point_cloud = point_cloud[:, :, : 3]
    centroids = t.zeros(batch_size, centroids_count, 3, dtype=point_cloud.dtype, device=device)
    distances = t.ones(batch_size, points, dtype=t.float, device=device) * float('inf')
    farthest = t.randint(0, points, (batch_size,), dtype=t.long, device=device)
    batch_indexes = t.arange(batch_size, dtype=t.long, device=device)

    for i in range(centroids_count):
        centroid = point_cloud[batch_indexes, farthest, :].view(batch_size, 3)

        centroids[:, i] = centroid
        distance = t.sum((point_cloud - centroid.view(batch_size, 1, 3)) ** 2, -1, dtype=t.float)
        querry = distance < distances
        distances[querry] = distance[querry]
        farthest = t.max(distances, -1)[1]

    return centroids


def sample_and_group(point_cloud, centroids_count, radius, samples_count):
    batch_size, cloud_size, dimensions = point_cloud.shape
    centroids = farthest_point_sampling(point_cloud, centroids_count)
    sampled_cloud = ball_query(point_cloud, centroids, radius, samples_count)

    if dimensions > 3:
        sampled_cloud_normalised = sampled_cloud[:, :, :, : 3] - centroids.view(batch_size, centroids_count, 1, 3)
        sampled_cloud_normalised = t.cat((sampled_cloud_normalised, sampled_cloud[:, :, :, 3 : ]), dim=-1)
    else:
        sampled_cloud_normalised = sampled_cloud - centroids.view(batch_size, centroids_count, 1, dimensions)

    return centroids, sampled_cloud_normalised


def sample_and_group_all(point_cloud):
    batch_size, cloud_size, dimensions = point_cloud.shape
    device = point_cloud.device

    centroids = t.zeros(batch_size, 1, 3, device=device)
    sampled_cloud = point_cloud.view(batch_size, 1, cloud_size, dimensions)

    return centroids, sampled_cloud


def main():
    a = [[[0, 0, 0, 0], [0, 1, 0, 1],
          [0, 0, 1, 2], [1, 0, 0, 3],
          [5, 0, 0, 4], [0, 5, 0, 5],
          [0, 0, 5, 6], [3, 4, -1, 7],
          [20, 20, 20, 8], [20, 20, 21, 9],
          [20, 21, 20, 10], [21, 20, 20, 11]],

         [[0, 0, 0, 0], [1, 1, 0, 1],
          [0, 1, 1, 2], [1, 1, 1, 3],
          [20, 20, 20, 4], [20, 20, 21, 5],
          [19, 18, 20, 6], [20, 30, 20, 7],
          [120, 120, 120, 8], [120, 120, 121, 9],
          [119, 118, 120, 10], [120, 130, 120, 11]],

         [[0, 0, 0, 0], [-1, -1, 0, 1],
          [0, -1, -1, 2], [-1, -1, -1, 3],
          [20, 20, 20, 4], [20, 20, 21, 5],
          [19, 18, 20, 6], [20, 30, 20, 7],
          [-120, -120, -120, 8], [-120, -120, -121, 9],
          [-119, -118, -120, 10], [-120, -130, -120, 11]]]

    pc = t.tensor(a, dtype=t.float)
    print(pc)
    d = farthest_point_sampling(pc, 3)
    print(d)
    print(ball_query(pc, d, 2, 2))
    print(sample_and_group(pc, 2, 5, 3))
    print(sample_and_group(pc, 2, 5, 3)[1].shape)
    print(sample_and_group_all(pc))
    print(sample_and_group_all(pc)[1].shape)


if __name__ == '__main__':
    main()
