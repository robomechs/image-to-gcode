## image-to-gcode
# Image-to-gcode for windows and linux without linuxcnc
The program converts Z-depth map image (.png .jpg .gif) to the Gcode for 2.5D milling.  
![2018-11-23_01-30-13](https://user-images.githubusercontent.com/8062959/48923446-a89fa880-eebf-11e8-8d07-e4cf2a294145.png)  
run image-to-gcode.ry to start the program

# Settings description
*Some description is copied from here: http://www.linuxcnc.org/docs/2.4/html/gui_image-to-gcode.html*
* **Units** Specifies whether to use G20 (inches) or G21 (mm) in the generated g-code and as the units for each option labeled (units). 
* **Invert Image** If “no”, the black pixel is the lowest point and the white pixel is the highest point. If “yes”, the black pixel is the highest point and the white pixel is the lowest point.
* **Normalize Image** If “yes”, the darkest pixel is remapped to black, the lightest pixel is remapped to white.
* **Extend Image Border** If “None”, the input image is used as-is, and details which are at the very edges of the image may be cut off. If “White” or “Black”, then a border of pixels equal to the tool diameter is added on all sides, and details which are at the very edges of the images will not be cut off.
* **Tolerance (units)** When a series of points are within **tolerance** of being a straight line, they are output as a straight line. Increasing tolerance can lead to better contouring performance in emc, but can also remove or blur small details in the image.
* **Pixel Size (Units)** One pixel in the input image will be this many units--usually this number is much smaller than 1.0. For instance, to mill a 2.5x2.5-inch object from a 400x400 image file, use a pixel size of .00625, because 2.5 / 400 = .00625.
* **Feed Rate (units per minute)** The feed rate for other parts of the path
* **Plunge Feed Rate (units per minute)** The feed rate for the initial plunge movement
* **Spindle Speed (RPM)**
* **Scan Pattern** Possible scan patterns are:
    - Rows
    - Columns
    - Rows, then Columns
    - Columns, then Rows
    - Rows Object
    - Columns Object
* **Scan Direction**
    - Positive: Start milling at a low X or Y axis value, and move towards a high X or Y axis value
    - Negative: Start milling at a high X or Y axis value, and move towards a low X or Y axis value
    - Alternating: Start on the same end of the X or Y axis travel that the last move ended on. This reduces the amount of traverse movements
    - Up Milling: Start milling at low points, moving towards high points
    - Down Milling: Start milling at high points, moving towards low points
* **Depth (units)** The top of material is always at Z=0. The deepest cut into the material is Z=-depth.
* **Vertical border for cutting background (units, 0=no cutting)**
* **Horizontal 'Max background len.' (pixels)**
* **Stepover (pixels)** The distance between adjacent rows or columns. To find the number of pixels for a given units distance, compute distance/pixel size and round to the nearest whole number. For example, if pixel size=.006 and the desired step over distance=.015, then use a Step Over of 2 or 3 pixels, because .015/.006=2.5.
* **Safety Height (units)** The height to move to for traverse movements. image-to-gcode always assumes the top of material is at Z=0.
* **Tool Diameter (units)** The diameter of the cutting part of the tool.
* **Tool Type** The shape of the cutting part of the tool. Possible tool shapes are:
    - Ball End
    - Flat End
    - 30 degree
    - 45 degree
    - 60 degree
    - 90 degree
* **Tool Diameter 2 (units)**
* **Angle of tool 2**  
![tool](https://user-images.githubusercontent.com/8062959/48924304-9aee2100-eec7-11e8-9b97-985f5bf5ba8e.jpg)  
* **Lace Bounding** This controls whether areas that are relatively flat along a row or column are skipped. This option only makes sense when both rows and columns are being milled. Possible bounding options are:
    - None: Rows and columns are both fully milled.
    - Secondary: When milling in the second direction, areas that do not strongly slope in that direction are skipped.
    - Full: When milling in the first direction, areas that strongly slope in the second direction are skipped. When milling in the second direction, areas that do not strongly slope in that direction are skipped.
* **Contact Angle (degrees)** When **Lace bounding** is not None, slopes greater than **Contact angle** are considered to be “strong” slopes, and slopes less than that angle are considered to be weak slopes.
* **Mill layer by layer** 
* **Does not cut on the passed(vertical optimize path)** 
* **Previous cutter minus current cutter(RMF)** 
* **Min delta of RMF mode (units)** 
* **Previous offset (rmf)(units, 0=no roughing)** 
* **Roughing offset (units, 0=no roughing)** Image-to-gcode can optionally perform rouging passes. The depth of successive roughing passes is given by “Roughing depth per pass”. For instance, entering 0.2 will perform the first roughing pass with a depth of 0.2, the second roughing pass with a depth of 0.4, and so on until the full Depth of the image is reached. No part of any roughing pass will cut closer than Roughing Offset to the final part.
* **Roughing depth per pass (units)**  
![roughingoffset_sm](https://user-images.githubusercontent.com/8062959/48924090-9e80a880-eec5-11e8-8608-a015afdc7cf2.jpg)  
* **Previous stepover (pixels)** 
* **Previous tool Diameter (units)** 
* **Previous tool Type** 
* **Previous tool Diameter 2 (units)** 
* **Previous Angle of tool diameter 2** 
* **Cut top jumper** 
* **The detail of comments** 

# About
This repository is a development of a well-known Linuxcnc application: [image-to-gcode](http://www.linuxcnc.org/docs/2.4/html/gui_image-to-gcode.html).  
Also, this project was developed by the author under the nickname [Harmonist](http://www.cnc-club.ru/forum/viewtopic.php?t=3541).  
However, until the last moment, the program remained integrated with "Linuxcnc".
This circumstance did not allow a wide range of users to apply it to their needs.
Current version is designed to eliminate this small flaw.
Now you can use it on Windows and Linux as a stand-alone program.  
You should use **Python 2.7** with **numpy**, **pillow** and Tkinter packages.

# License
image-to-gcode is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 2 of the License, or (at your
option) any later version.  image-to-gcode is distributed in the hope 
that it will be useful, but WITHOUT ANY WARRANTY; without even the implied
warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See
the GNU General Public License for more details.  You should have
received a copy of the GNU General Public License along with image-to-gcode;
if not, write to the Free Software Foundation, Inc., 59 Temple Place,
Suite 330, Boston, MA 02111-1307 USA

# Authors
image-to-gcode.py is Copyright (C) 2005 Chris Radek  
chris@timeguy.com  
image-to-gcode.py is Copyright (C) 2006 Jeff Epler  
jepler@unpy.net  
image-to-gcode.py is Copyright (C) 2013 Harmonist  
cnc-club.ru  
image-to-gcode.py is Copyright (C) 2018 Yaroslav Vlasov  
ysvlasov@yandex.ru  
