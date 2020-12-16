# MCNCC - Multichannel normalized cross correlation

This script uses mutliple features channels followed subsequently with a calculation of the cross correlation between two images in order to find the pair with the highest chance of having the same source. The original purpose of this script was to find matching shoe impressions from crime scenes and clean reference images from an existing database (FID-300). It is possible to use this script for a variety of tasks with similar purpose, though if you want to create a cmc-diagram you will need a csv-file with the matching pairs (an example is provided in the repository, called Subsetlabels.csv).



## Step by step intruction of the script:

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
     python main.py -f path/to/your/project/folder -t path/to/your/track/folder -rf path/to/your/reference/folder
```

optional arguments:

  *-h*, --help            show this help message and exit
  
  *-f* FOLDER, --folder FOLDER
  <br />
  define folder containing the dataset
                        
  -t TRACKS, --tracks TRACKS
                        define track folder
                        
  -rf REFS, --refs REFS
                        define reference folder
                        
  -str STRIDE, --stride STRIDE
                        stride for convolutions
                        
  -avgp, --avgpool_bool
                        activate average pooling for features
                        
  -avgp_str AVGP_STRIDE, --avgp_stride AVGP_STRIDE
                        stride for average_pooling
                        
  -skf, --skip_feat     skip feature generation
  
  -r, --rot             add rotation
  
  -ris START, --start START
                        rotation interval start
                        
  -rie END, --end END   rotation interval end
  
  -sf SCOREFILE, --scorefile SCOREFILE
                        scorefilename
                        
  -cmc, --cmc           calculate cmc
  
  -cmcf CMC_FILE, --cmc_file CMC_FILE
                        cmc filename
                        
  -lbltable LABEL_FILE, --label_file LABEL_FILE
                        name of the csv. file

                        

- after running the program, a .npy file is created storing the correlation matrix (rows: number of tracks in the chosen track folder, columns: number of reference images in the chosen reference image folder)

- use the cmc argument in order to create cmc-plots from your correlation score-files

This function creates for example following graph:
```
     python -f path/to/your/project/folder -t path/to/your/track/folder -rf path/to/your/reference/folder 
```
<img src="cmc_score_diagram.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />    

- If you don't have access to a GPU you can also use google collaboratory through this link in order to test out the algorithm:
https://drive.google.com/drive/folders/13txeoZfnQ6rAHktlV3-q9x69nJ-rg8qt?usp=sharing
