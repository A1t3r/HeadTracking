# HeadTracking

## How to run:
 1. install requirements via pip install -r requirements.txt
 2. run run_track.py with specified parameters which are described below

parameters:
 <br />   -path - path where MOT data is stored
 <br />  -vout - path where video out will be stored
 <br />  -tout - path where tracks out will be stored
 <br />   -cdc - coef for color hist distance (from 0 to 1). If 0 is specified, so no color histogram similarity will be used
 <br />  -tracker - if you want to use BYTE tracker specify 'byte', in other cases DeepSORT will be used
