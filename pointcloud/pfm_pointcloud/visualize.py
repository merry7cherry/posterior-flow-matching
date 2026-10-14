"""Simple point-cloud gallery."""
import math


def save_plot(points, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    count = min(8, len(points))
    columns = min(4, count)
    rows = math.ceil(count / columns)
    fig = plt.figure(figsize=(3 * columns, 3 * rows))
    for index, cloud in enumerate(points[:count].cpu().numpy()):
        ax = fig.add_subplot(rows, columns, index+1, projection='3d')
        ax.scatter(cloud[:,0], cloud[:,2], cloud[:,1], s=0.3)
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=20, azim=35)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
