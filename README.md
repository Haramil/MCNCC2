# MCNCC - Multichannel normalized cross correlation

## Step by step intruction:

- Download this repository into your desired location via:
```
     git clone https://github.com/ErikFaustmann/MCNCC2.git
```
You can add you own dataset to your project folder, but keep in mind to provide a label table (csv) in order to plot a cmc diagram.

- Create a virtual environment within this folder:
```
     python -m venv mcncc_venv
```

- Install all the required libraries within this environment via the requirements.txt file

- Start Skript:

Example (strides 2, rotation acticated and cmc-score output)

```
     python main.py -f path/to/your/project/folder -t path/to/your/track/folder -rf path/to/your/reference/folder -str 2 -r -cmc
```
