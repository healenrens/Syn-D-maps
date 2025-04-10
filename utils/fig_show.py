import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
def fig_show(data_before, data_after, save_path):
    colors = {
        'road_divider': 'red',
        'lane_divider': 'orange',
        'ped_crossing': 'blue',
        'road_segment': 'black',
        'lane': 'green',
        'road_boundary': 'purple',
         'null_line': 'black'
    }
    plt.figure(figsize=(12, 12))
    for key, segments in data_before.items():
        for segment in segments:
            if len(segment)!=0:
                x, y = zip(*segment)
                plt.plot(x, y, label=key, color=colors[key], alpha=0.5)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("Visualization of Road Data - Before")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(save_path+'_before.jpg')
    plt.close()
    plt.figure(figsize=(12, 12))
    for key, segments in data_after.items():
        if key == 'lane':
            continue
        for segment in segments:
            if len(segment)!=0:
                x, y = zip(*segment)
                plt.plot(x, y, label=key, color=colors[key], alpha=0.7)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.title("Visualization of Road Data - After")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(save_path+'_after.jpg')
    plt.close()

