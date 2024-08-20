# LOID: Lane Occlusion Inpainting and Detection for Enhanced Autonomous Driving Systems

## Overview

LOID (Lane Occlusion Inpainting and Detection) is a project aimed at enhancing autonomous driving systems by effectively detecting and inpainting occluded lane markings. This system ensures more reliable lane detection even in scenarios where lanes are partially obscured due to vehicles, debris, or other obstacles, improving overall driving safety.

## Features

- **Lane Occlusion Detection**: Identifies areas where lane markings are occluded.
- **Inpainting Module**: Reconstructs occluded parts of the lane, ensuring continuity in lane detection.
- **Enhanced Lane Detection**: Integrates inpainted lanes into the overall lane detection pipeline for more robust performance.

## Installation

### Prerequisites

- **Python**: Version 3.8.10
- **CUDA**: Version 12.1

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/LOID.git
   cd LOID```
2. **Initialize the environment:**
Ensure you have Python 3.8.10 installed and CUDA 12.1 properly configured.
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
4. **Download the pretrained weights:**
Link: https://drive.google.com/file/d/1ruQaSMieBwh--SuvuF6XICDzR_py1DrW/view?usp=sharing
extract the weights folder in the repository
### Running the code
To run the project, simply execute the test.sh script in the project directory:
``` bash
./test.sh
```
### Modifying variables
If you need to adjust variables or configurations, modify the test.sh script according to your needs.

### License
This project is licensed under the MIT License - see the LICENSE file for details





