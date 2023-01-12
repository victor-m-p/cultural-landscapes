final parameter fits

* use cleaned* for all analyses (these are the ones with texts removed)

* cleaned_* is the data; reference_with_entry_id_cleaned_* has the entry_ids listed

* based on the lists that appear in /data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv

* questions have the same order as in /data/reference/main_nrow_660_ncol_21_nuniq_20_suniq_581_maxna_10.csv

* the following entry entry ids have been removed, because they are tagged by DRH as religious texts
[1025, 1042, 1050, 1054, 1055, 1056, 1057, 1058, 1062, 1063, 1067, 1074, 1075, 1078, 1079, 1080, 1098, 1100, 1112, 1117, 1120, 1121, 1124, 1141, 1144, 1158, 1160, 1170, 1176, 1177, 1182, 1188, 1203, 1207, 1224, 1232, 1239, 1243, 1252, 1253, 1255, 1260, 1265, 1269, 1270, 1271, 1272, 1290, 1294, 1297, 1324, 1326, 1328, 1330, 1338, 1342, 1345, 1347, 1356, 1367, 1375, 1387, 1389, 1393, 1401, 1405, 1406, 1417, 1418, 1421, 1422, 1424, 1427, 1429, 1432, 1442, 1443, 1452, 1471, 1473, 1484, 1486, 1508, 1509, 1513, 1525]

* the fits are with one nearest neighbour, and have the suffix _params.dat; J's and then h's order.

* right now, we have NOT characterized the stability for N=20 when NA > 5; we have the CPU-hours, so I've run the fits on the Pittsburgh Supercomputer Center for NAs larger, just so we have them.
