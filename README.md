## image-to-gcode
# Image-to-gcode for windows and linux without linuxcnc
The program converts Z-depth map image (.png .jpg .gif) to the Gcode for 2.5D milling.
![2018-11-23_01-30-13](https://user-images.githubusercontent.com/8062959/48923446-a89fa880-eebf-11e8-8d07-e4cf2a294145.png)

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
