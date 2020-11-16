<h1> Calculating the number π using the Monte Carlo method </h1>
 
This program allows you to calculate the number of pi using the CPU and GPU using the Monte Carlo method the results in the following table.
 
<html>
<head>
<meta charset="utf-8">
</head>
<body>
<table>
<tr>
<th>Number of points</th>
<th>CPU Time</th>
<th>GPU Time</th>
<th>CPU Time/ GPU Time</th>
<th> pi value </th>
</tr>
<tr><td>65536</td> <td>0.0557 s</td> <td>0.0010 s</td> <td>55.700</td> <td>3.1418</td></tr>
<tr><td>65536*2</td> <td>0.0865 s</td> <td>0.0014 s</td> <td>61.7857</td> <td>3.1576</td></tr>
<tr><td>65536*4</td> <td>0.1918 s</td> <td>0.00163 s</td> <td>117.6683 </td> <td>3.1491</td></tr>
<tr><td>65536*8</td> <td>0.3816 s</td> <td>0.0027 s</td> <td>141.333</td> <td>3.1423</td></tr>
<tr><td>65536*16</td> <td>0.7751 s</td> <td>0.0033 s</td> <td>234.8776</td> <td>3.1455</td></tr>
</table>
</body>
</html>
