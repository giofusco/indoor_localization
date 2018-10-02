function plot_altimeter_data(vio_file1, vio_file2, vio_file3)

[t1, M, coord1, rot, alt1, vio_status] = vio_parser(vio_file1, 1);
[t2, M, coord2, rot, alt2, vio_status] = vio_parser(vio_file2, 1);
[t3, M, coord3, rot, alt3, vio_status] = vio_parser(vio_file3, 1);
tstamp1 = cell2mat(t1{1,1})';
pressure1 = cell2mat(alt1{1,1})';

tstamp2 = cell2mat(t2{1,1})';
pressure2 = cell2mat(alt2{1,1})';

tstamp3 = cell2mat(t3{1,1})';
pressure3 = cell2mat(alt3{1,1})';

c1 = cell2mat(coord1{1,1})';
c2 = cell2mat(coord2{1,1})';
c3 = cell2mat(coord3{1,1})';

figure, plot(tstamp1 - tstamp1(1), pressure1 - pressure1(1))
hold on, plot(tstamp2 - tstamp2(1), pressure2- pressure2(1))
hold on, plot(tstamp3 - tstamp3(1), pressure3- pressure3(1))
% 
% hold on, plot(tstamp1 - tstamp1(1), c1(:,2) - c1(1,2))
% hold on, plot(tstamp2 - tstamp2(1), c2(:,2) - c2(1,2))
% hold on, plot(tstamp3 - tstamp3(1), c3(:,2) - c3(1,2))

% figure, plot(c1(:,2) - c1(1,2), c1(:,2) - c1(1,2))
% hold on, plot(c2(:,2) - c2(1,2), c2(:,2) - c2(1,2))
% hold on, plot(c3(:,2) - c3(1,2), c3(:,2) - c3(1,2))
