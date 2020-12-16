# MCNCC - Multichannel normalized cross correlation

This script uses mutliple features channels followed subsequently with a calculation of the cross correlation between two images in order to find the pair with the highest chance of having the same source. The original purpose of this script was to find matching shoe impressions from crime scenes and clean reference images from an existing database (FID-300). It is possible to use this script for a variety of tasks with similar purpose, though if you want to create a cmc-diagram you will need a csv-file with the matching pairs (an example is provided in the repository, called Subsetlabels.csv).



## Step by step intruction of the script:

### Set up

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

### Starting script

Example (strides 2, rotation acticated and cmc-score output)
```
     python main.py -f path/to/your/project/folder -t path/to/your/track/folder -rf path/to/your/reference/folder
```

optional arguments:

  *-h*, --help            show this help message and exit
  
  *-f* <b>FOLDER</b>, --folder <b>FOLDER</b>
  <br />
  define folder containing the dataset
                        
  -t <b>TRACKS</b>, --tracks <b>TRACKS</b>
  <br />
  define track folder
                        
  -rf <b>REFS</b>, --refs <b>REFS</b>
  <br />define reference folder
                        
  -str <b>STRIDE</b>, --stride <b>STRIDE</b>
  <br />stride for convolutions
                        
  -avgp, --avgpool_bool
  <br/>activate average pooling for features
                        
  -avgp_str <b>AVGP_STRIDE</b>, --avgp_stride <b>AVGP_STRIDE</b>
  <br />stride for average_pooling
                        
  -skf, --skip_feat     
  <br />skip feature generation
  
  -r, --rot             
  <br />add rotation
  
  -ris <b>START</b>, --start <b>START</b>
  <br />rotation interval start
                        
  -rie <b>END</b>, --end <b>END</b>  
  <br />rotation interval end
  
  -sf <b>SCOREFILE</b>, --scorefile <b>SCOREFILE</b>
  <br />scorefilename
                        
  -cmc, --cmc           
  <br />calculate cmc
  
  -cmcf <b>CMC_FILE</b>, --cmc_file <b>CMC_FILE</b>
  <br />cmc filename
                        
  -lbltable <b>LABEL_FILE</b>, --label_file <b>LABEL_FILE</b>
  <br />name of the csv. file

                        

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
