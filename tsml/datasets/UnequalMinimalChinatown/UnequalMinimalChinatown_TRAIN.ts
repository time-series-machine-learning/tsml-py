% This is a cut down version of the problem ChinaTown, useful for code examples and unit tests
% http://www.timeseriesclassification.com/description.php?Dataset=Chinatown
% The train set is the same as ChinaTown, but the test set is reduced from 345 cases to 20 cases
% To make the series unequal length, values have arbitratily been removed from the beginning and end of series
%
@problemName UnequalMinimalChinatown
@timeStamps false
@missing false
@univariate true
@equalLength false
@classLabel true 1 2
@data
212.0,55.0,34.0,25.0,33.0,113.0,143.0,303.0,615.0,1226.0,1281.0,1221.0,1081.0,866.0,1096.0,1039.0,975.0,746.0,581.0,409.0,182.0:1
104.0,28.0,28.0,25.0,70.0,153.0,401.0,649.0,1216.0,1399.0,1249.0,1240.0,1109.0,1137.0,1290.0,1137.0,791.0,638.0,597.0,316.0:1
603.0,348.0,176.0,177.0,47.0,30.0,40.0,42.0,101.0,180.0,401.0,777.0,1344.0,1573.0,1408.0,1243.0,1141.0,1178.0,1256.0,1114.0,814.0,635.0,304.0,168.0:1
428.0,309.0,199.0,117.0,82.0,43.0,24.0,64.0,152.0,183.0,408.0,797.0,1288.0,1491.0,1523.0,1460.0,1365.0,1520.0,1700.0,1797.0:1
372.0,310.0,203.0,133.0,65.0,39.0,27.0,36.0,107.0,139.0,329.0,651.0,990.0,1027.0,1041.0,971.0,1104.0,844.0,1023.0,1019.0,862.0,643.0,591.0,452.0:1
448.0,344.0,183.0,146.0,71.0,14.0,30.0,41.0,108.0,137.0,277.0,576.0,1010.0,1271.0,1264.0,1062.0,1093.0,1030.0,1069.0,1151.0,898.0,754.0,467.0,362.0:1
621.0,322.0,221.0,150.0,65.0,40.0,42.0,84.0,148.0,190.0,341.0,685.0,1162.0,1391.0,1367.0,1279.0,1318.0,1336.0,1440.0,1479.0:1
597.0,409.0,142.0,93.0,48.0,30.0,34.0,87.0,132.0,157.0,389.0,1024.0,1648.0,1768.0,1703.0,1706.0,1520.0,1562.0,1608.0,1766.0:1
525.0,431.0,248.0,240.0,91.0,64.0,29.0,117.0,200.0,236.0,456.0,717.0,1331.0,1609.0,1563.0,1398.0,1465.0,1459.0,1631.0,1891.0,1847.0,1731.0,1375.0,1188.0:1
192.0,130.0,44.0,21.0,35.0,73.0,132.0,299.0,639.0,1110.0,1320.0,1208.0,1069.0,1070.0,1164.0,1547.0,1575.0,1503.0,1139.0,1066.0,776.0:1
144.0,73.0,21.0,16.0,10.0,12.0,26.0,100.0,177.0,220.0,371.0,599.0,1280.0,1346.0,1461.0,1367.0,1319.0,1205.0,1250.0,1269.0,981.0,806.0,550.0,320.0:2
141.0,63.0,51.0,14.0,16.0,14.0,28.0,103.0,172.0,205.0,442.0,601.0,1165.0,1228.0,912.0,872.0,847.0,1227.0,1561.0,1611.0,1386.0,1007.0,810.0,680.0:2
67.0,67.0,107.0,14.0,11.0,19.0,18.0,75.0,185.0,223.0,298.0,571.0,1044.0,1107.0,872.0,765.0,824.0,1032.0,1237.0,1084.0,759.0,543.0,377.0,168.0:2
69.0,54.0,35.0,6.0,7.0,15.0,19.0,76.0,212.0,238.0,364.0,717.0,1223.0,1166.0,994.0,931.0,929.0,1093.0,1275.0,1161.0,1083.0,726.0,441.0,241.0:2
142.0,104.0,55.0,33.0,51.0,13.0,36.0,77.0,185.0,222.0,437.0,739.0,1326.0,1372.0,979.0,942.0,1019.0,1250.0,1663.0,1781.0,1513.0:2
256.0,171.0,104.0,67.0,51.0,26.0,25.0,41.0,170.0,192.0,334.0,801.0,1341.0,1468.0,1395.0,1221.0,1168.0,1284.0,1400.0,1321.0,1099.0,791.0,498.0,272.0:2
10.0,16.0,28.0,128.0,226.0,271.0,402.0,652.0,1282.0,1353.0,1084.0,1124.0,920.0,1063.0,1160.0,1184.0,893.0,720.0,541.0,304.0:2
78.0,60.0,51.0,17.0,10.0,12.0,22.0,73.0,172.0,186.0,318.0,414.0,1003.0,1153.0,981.0,846.0,872.0,1051.0,1200.0,1276.0,1004.0,841.0,525.0,315.0:2
26.0,3.0,23.0,28.0,108.0,209.0,241.0,372.0,549.0,1206.0,1223.0,1156.0,1102.0,1083.0,1107.0,1109.0,1193.0,900.0,660.0,442.0,226.0:2
28.0,39.0,17.0,30.0,138.0,225.0,261.0,358.0,621.0,1074.0,1176.0,914.0,858.0,830.0,928.0,954.0,871.0,730.0,506.0,262.0,150.0:2
