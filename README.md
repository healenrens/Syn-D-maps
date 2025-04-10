# Syn-D-Maps

Syn-D-Maps is a comprehensive toolkit for transforming and augmenting map data from the nuScenes dataset. The library enables various geometric and semantic modifications to road networks, making it ideal for generating diverse map variants for autonomous driving research, data augmentation, and testing. Due to the current grouping scheme still having instability in some cases, the results may sometimes deviate. We will continue to optimize our code.

## Features

### 1. Topological Transformations (`top_utils_lib`)

- **Lane Operations**:
  - `add_lane`: Add new lanes to the map
  - `delete_lane`: Remove existing lanes
  - `lane_narrowing`: Decrease lane width
  - `lane_widening`: Increase lane width
- **Road Structure Transformations**:
  - `delete_group`: Remove groups of related map elements
  - `divider_delete`: Delete road or lane dividers
  - `bezier_warp`: Apply bezier curve deformation to road elements

### 2. Semantic Transformations (`seg_utils_lib`)

- `semanteme_change`: Change semantic properties of map elements
- `semanteme_delete`: Remove specific semantic elements from the map

### 3. Geometric Transformations (`geo_utils_lib`)

- `global_rotate`: Rotate all map elements around a center point
- `global_translate`: Apply translation to all map elements
- `random_global_translate`: Apply random translation with controlled distance

## Installation

### Prerequisites

- Python 3.7+
- nuScenes devkit
- NumPy
- Matplotlib

### Setup

1. Clone the repository:

```bash
git clone https://github.com/healenrens/Syn-D-maps
cd Syn-D-maps
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download and set up the nuScenes dataset following the [official instructions](https://github.com/nutonomy/nuscenes-devkit).

## Usage

The basic workflow involves:

1. Loading map data from nuScenes
2. Extracting map geometries
3. Applying one or more transformations
4. Visualizing or saving the results

### Basic Example

For a complete example with all transformation types, see [example.py](example.py).

## Transformation Parameters

Most transformation functions accept common parameters:

- `all_line`: Dictionary containing map geometries
- `rate`: Transformation intensity (usually 0-1)
- `distance_level`: For geometric transformations ('low', 'medium', 'high')
- Additional transformation-specific parameters

## License

[MIT License](LICENSE)

## Citation

If you use Syn-D-maps in your research, please cite:

```
@misc{syn-d-maps,
  author = {Haoming Xu, Yiyang Xiao, Wei Li, Yu Hu},
  title = {Generating Synthetic Deviation Maps for Prior-Enhanced Vectorized HD Map Construction},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/healenrens/Syn-D-maps}}
}
```

References to our paper will be released after publication.
