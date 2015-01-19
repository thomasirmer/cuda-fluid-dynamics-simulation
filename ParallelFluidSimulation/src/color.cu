/******************************************************************************
 This function converts HSV values to RGB values, scaled from 0 to maxBrightness

 The ranges for the input variables are:
 hue: 0-360
 sat: 0-255
 lig: 0-255

 The ranges for the output variables are:
 r: 0-maxBrightness
 g: 0-maxBrightness
 b: 0-maxBrightness

 r,g, and b are passed as pointers, because a function cannot have 3 return variables
 Use it like this:
 int hue, sat, val;
 unsigned char red, green, blue;
 // set hue, sat and val
 hsv2rgb(hue, sat, val, &red, &green, &blue, maxBrightness); //pass r, g, and b as the location where the result should be stored
 // use r, b and g.

 (c) Elco Jacobs, E-atelier Industrial Design TU/e, July 2011.

 *****************************************************************************/


