clear all
[t, M, coord, rot, alt, vio_status] = vio_parser('./alt_data/VIO_down_up.txt', 1);
close all
tstamp = cell2mat(t{1,1})';
pressure = cell2mat(alt{1,1})'*10; % convert from kPA to hPA

% if too many samples have zero delta, then we should make a decision 
zero_count_threshold = 20; % about 2 seconds

figure, plot(tstamp-tstamp(1), (pressure-pressure(1)))
% dT = diff(tstamp);
% dB = diff(B);
figure, plot(diff(pressure))
i_start = -1;
i_end = -1;
direction = 0;

hPA_threshold = 0.3;

prev_pressure = nan;

zero_count = 0;


for i = 1 : length(pressure)
    
    if isnan(prev_pressure)
        prev_pressure = pressure(i);
    else
        dPress = pressure(i) - prev_pressure;
        tmp_direction = 0;
        prev_pressure = pressure(i);
        % update direction of \Delta(Pressure)
        
        if dPress ~= 0 && i_start == -1
            i_start = i-1;
        end
        
        if dPress > 0
            tmp_direction = -1;
            zero_count = 0;
        elseif dPress < 0
            tmp_direction = 1;
            zero_count = 0;
        else
            zero_count = zero_count + 1;
        end
        
        % test if change in direction
        if direction * tmp_direction < 0 || zero_count > zero_count_threshold
            % run test
            i_end = i-2;
            zero_count = 0;
            if abs((pressure(i_end) - pressure(i_start))) > hPA_threshold
                i_start
                i_end
                disp('Detected change of elevation')
                disp(strcat('\Delta(T): ' , num2str(tstamp(i_end) - tstamp(i_start))))
                disp(strcat('\Delta(B): ', num2str(abs(pressure(i_end) - pressure(i_start)))))
                if direction == 1
                    disp('Going up');
                else
                    disp('Going down');
                end
            end
            
            direction = tmp_direction;
            i_start = i-1;   
            prev_pressure = pressure(i_start);
            
        elseif direction == 0 && tmp_direction ~= 0
            direction = tmp_direction;
        end
        
        
   
        
    end
    
    
end
