function plot_vio_data(vio_file, start_normal)

[t, M, coordinates, rot, vio_status] = vio_parser(vio_file)

    if nargin > 1
      start_normal = start_normal
    else
      start_normal = false;
    end

    coord = cell2mat(coordinates{1,1})';
    status = vio_status{1}';
    start_idx = 1;
    
    if start_normal == true
        % search for first normal status
        
        while ~strcmp(status(start_idx), 'normal')
            start_idx = start_idx + 1;
        end
    end

    deltas = diff(coord(start_idx:end,:));
    norms = sqrt(sum(deltas.^2,2))
    figure, 
    subplot(2,2,1), plot(coord(start_idx:end,3), coord(start_idx:end,1)),
    title(strcat('Total translation: ', ...
    num2str(norm(coord(end,:)))))
    subplot(2,2,2), plot(coord(start_idx+1:end,2)) , title('Y axis translation')
    subplot(2,2,3), plot(deltas(:,1))
    hold on, subplot(2,2,3), plot(deltas(:,3)), title('Position Deltas X and Z')
    hold on, subplot(2,2,4), plot(norms, '--X'), title('Deltas Norm (XYZ)')
end