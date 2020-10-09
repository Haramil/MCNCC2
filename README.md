# MCNCC - Multichannel normalized cross correlation

Step by step intruction:

Create a new project-folder, download this repository and extract the files into the new folder
Optinally you can add you own dataset to your project folder, but keep in mind to provide a label table (csv) in order to plot a cmc diagram.

Create a virtual environment with all the required libraries (requirements.txt)

Example (strides 2, rotation acticated and cmc-score output)

```
     python main.py -f path/to/your/project/folder -t path/to/your/track/folder -rf path/to/your/reference/folder -str 2 -r -cmc
```
