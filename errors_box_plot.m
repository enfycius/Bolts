%% CSV 파일에서 데이터 읽기
data = readtable('./errors.csv');

absolute_data = data(:, {'absolute_x', 'absolute_y'});
relative_data = data(:, {'relative_x', 'relative_y'});

figure;

yyaxis left;
boxplot([absolute_data.absolute_x, absolute_data.absolute_y], 'Labels', {'X', 'Y'}, 'Widths', 0.75, 'Positions', [1 2]);
ylabel('Absolute Error');

yyaxis right;
H = boxplot([relative_data.relative_x, relative_data.relative_y], 'Labels', {'X1', 'Y1'}, 'Widths', 0.75, 'Positions', [4 5]);
ylabel('Relative Error (%)');

xticks([1 2 4 5]);
xticklabels({'axis=0', 'axis=1', 'axis=0''', 'axis=1'''});

xlim([0, 8]);

title('Error on Distance between Bolt Holes');
xlabel('Categories');

grid on;

%% Relative Error Box Plots들에 Gray Color로 Fill
h = findobj(gca, 'Tag', 'Box');

grayShade = 0.5;
grayColor = [grayShade, grayShade, grayShade];

for i = 1:numel(h)-1
    if i == 3
        continue;
    end
    patch(get(h(i), 'XData'), get(h(i), 'YData'), grayColor, 'FaceAlpha', 0.5);
end

