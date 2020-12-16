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

Here is an example for the most simple command entry, that will calculate a correlation matrix and store the file within your main project folder (the first argument).
```
     python main.py -f path/to/your/project/folder -t path/to/your/track/folder -rf path/to/your/reference/folder
```

Optional arguments:

  *-h*, --help<br/>Show this help message and exit
  
  *-f* <b>FOLDER</b>, --folder <b>FOLDER</b><br/>Define the folder in which the main.py file is directly under.
                        
  *-t* <b>TRACKS</b>, --tracks <b>TRACKS</b><br/>Define track folder. The image files should be directly within this folder. 
                        
  *-rf* <b>REFS</b>, --refs <b>REFS</b><br/>Define the reference folder. The image files should be directly within this folder. 
                        
  *-str* <b>STRIDE</b>, --stride <b>STRIDE</b><br/>Stride for convolutions when calculating the cross correlation. A bigger stride leads to a decreasing calculation time (around 40% less), but can lead a slightly worse performance regarding the matching of images. The range of recommended strides goes from 1-4 for 200x300 sized images.
                        
  *-avgp*, --avgpool_bool<br/>Activate average pooling for feature generation. This can also lead to a decreased calculation time.
                        
  *-avgp_str* <b>AVGP_STRIDE</b>, --avgp_stride <b>AVGP_STRIDE</b><br/>Stride for average_pooling. The range of recommended strides goes from 1-2 for 200x300 sized images.
                        
  *-skf*, --skip_feat<br/>Skip feature generation if you already created features for your dataset. This reduces the overall calculation time for the script to execute.
  
  *-r*, --rot<br/>Enable rotation of track images. The goal is to increase the performance of your matching task, because the rotation of the track image might lead to a better alignment between the track image and the reference image. Keep in mind that if you activate rotation you need to also activate the ris, rie flags, explained below.
  
  *-ris* <b>START</b>, --start <b>START</b><br/>Rotation interval start: The starting angle for the track image, starting from a negativ number (for example -10) and zero being the original orientation.
                        
  *-rie* <b>END</b>, --end <b>END</b><br/>Rotation interval end: The ending angle for the track image, ending at a positive number (for example +10) and zero being the original orientation.
  
  *-sf* <b>SCOREFILE</b>, --scorefile <b>SCOREFILE</b><br/>The name or path (including the name) of the scorefile that is created after the script is finished. If there is no path given, it will be created directly within your main project folder.
                        
  *-cmc*, --cmc<br/>If you set this flag, a cmc calculation will be started. Keep in mind that you need to provide a label file within your project folder. An example is given in this repository named Subsetlabels.csv. 
  
  *-cmcf* <b>CMC_FILE</b>, --cmc_file <b>CMC_FILE</b><br/>The name or path (including the name) of the cmc png image that is created after the script is finished. If there is no path given, it will be created directly within your main project folder.
                        
  *-lbltable* <b>LABEL_FILE</b>, --label_file <b>LABEL_FILE</b><br/>Name of the csv. file containing the matching pairs of reference images and track images.


### Examples

This input creates for example following graph:
```
     python main.py -f path/to/your/project/folder -t path/to/your/track/folder -rf path/to/your/reference/folder 
```
<img src="cmc_score_diagram.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />    
     
Another example with average pooling:
```
     python main.py -f path/to/your/project/folder -t path/to/your/track/folder -rf path/to/your/reference/folder -avgp -avgp_str 2 -cmc
```
<img src="cmc_score_diagram.png"
     alt="Markdown Monster icon"
     style="float: left; margin-right: 10px;" />   

### Additional remarks

If you don't have access to a GPU you can also use google collaboratory through this link in order to test out the algorithm:
https://drive.google.com/drive/folders/13txeoZfnQ6rAHktlV3-q9x69nJ-rg8qt?usp=sharing
