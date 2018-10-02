function [t, M, coord, rot, alt, vio_status] = vio_parser_2(vio_filename, altimeter)

fileID = fopen(fullfile(vio_filename),'r');
do_read = 1;
matrix_line = fgetl(fileID); %skip header
cnt = 1;
while do_read
    timestamp_line = fgetl(fileID);
    status_line = fgetl(fileID);
    matrix_line = fgetl(fileID);
    coord_line = fgetl(fileID);
    angles_line = fgetl(fileID);
    if altimeter == 1
        pressure_line = fgetl(fileID);
    end

    if (status_line == -1)
        do_read = 0;
        break;
    end
    vio_status{cnt} = status_line;
    t(cnt, :) = str2double(timestamp_line);
    M{cnt} = parse_matrix_string(matrix_line);
    coord(cnt, :) = parse_coordinate_string(coord_line);
    rot(cnt, :) = parse_angles_line(angles_line);
    if altimeter == 1
        alt(cnt, :) = str2double(pressure_line);
    end
    cnt = cnt + 1;

end

end


function coord = parse_coordinate_string(s)
formatSpec = '%f,%f,%f';

coord = sscanf(s, formatSpec, 3);
end

function v = parse_angles_line(l)
formatSpec = '%f,%f,%f';
pos = strfind(l,'(');
v = sscanf(l(pos(1)+1:end), formatSpec, 3);
end

function M = parse_matrix_string(m)
formatSpec = '%f,%f,%f,%f';
pos = strfind(m,'[');
M = zeros(4,4);
l1 = sscanf(m(pos(2)+1:end), formatSpec, 4);
l2 = sscanf(m(pos(3)+1:end), formatSpec, 4);
l3 = sscanf(m(pos(4)+1:end), formatSpec, 4);
l4 = sscanf(m(pos(5)+1:end), formatSpec, 4);
M(1,:) = l1;
M(2,:) = l2;
M(3,:) = l3;
M(4,:) = l4;
end
