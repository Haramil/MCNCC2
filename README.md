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

- Within your project folder you can  now activate your virtual environment via:
```
     mcncc_venv\Scripts\activate
```

- Install all the required libraries within this environment via the provided requirements.txt file:
```
     pip install -r requirements.txt
```
- Additionally you will need to install torch. Go to the website and follow the guidelines:
https://pytorch.org/

- Start Skript:

Example (strides 2, rotation acticated and cmc-score output)
```
     python main.py -f path/to/your/project/folder -t path/to/your/track/folder -rf path/to/your/reference/folder -str 2 -r -cmc
```
