[t, M, coord, rot, alt, vio_status] = vio_parser('./alt_data/VIO_up_1_stairs_1.txt', 1);
% close all
tstamp = cell2mat(t{1,1})';
pressure = cell2mat(alt{1,1})';

B = (pressure-pressure(1))*10;
% B = movmean(pressure*10, 11);
figure, plot(tstamp-tstamp(1), B)
% figure, plot(B(1:5:end))
dT = diff(tstamp);

dB = diff(B);
figure, plot(tstamp(1:end-1)-tstamp(1), dB)
% findpeaks(dB)
% hold on, plot(tstamp-tstamp(1), pressure-pressure(1))

peakStart = -1;
peakEnd = -1;

for i = 1 : length(dB)-1
    
    % is it a peak
    if dB(i) > 0
        % special case: first value is > 0, start a peak
        if i == 1
            peakStart = i;
        else
            % are we continuing a peak?
            if dB(i-1) <= 0
                %analyze previous sequence?
                if peakStart ~= -1 && peakEnd ~= -1
                    area = trapz(dT(peakStart:peakEnd), dB(peakStart:peakEnd))
                    if abs(area) >= 1
                        disp('HEY!')
                    end
                    % do something about it
                end              
                peakStart = i-1;
                peakEnd = -1;
            else
                peakEnd = i+1;
            end
        end
    end
    
    %is it a through?
    if dB(i) < 0
        % special case: first value is < 0, start a through
        if i == 1
            peakStart = i;
        else
            % are we continuing a peak?
            if dB(i-1) >= 0
                %analyze previous sequence?
                if peakStart ~= -1 && peakEnd ~= -1
                    area = trapz(tstamp(peakStart+1:peakEnd+1), dB(peakStart:peakEnd))
                    % do something about it
                    if abs(area) >= 1
                        disp('HEY 2!')
                    end
                end              
                peakStart = i;
                peakEnd = -1;
            else
                peakEnd = i;
            end
        end
    else
        if peakStart ~= -1 && peakEnd ~= -1
            area = trapz(tstamp(peakStart+1:peakEnd+1), dB(peakStart:peakEnd))
            % do something about it
            if abs(area) >= 1
                disp('HEY 3!')
            end
            peakStart = -1;
            peakEnd = -1;
        end
    end

end
   
    