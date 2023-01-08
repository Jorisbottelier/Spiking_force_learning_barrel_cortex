% Creates plots for the initialized weight matrices.

% For 1 < G < 5:

% G_vals = 1:0.5:5;
% for i=1:length(G_vals)
%      filename = ['Win_1G_' num2str(G_vals(i)) 'Q_1Winp_1Pexc_0.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      disp(training_output.acc);
%      subplot(3, 3, i);
%      imagesc(training_output.weights.static(50:100, 50:100));
%      axis square;
%      h.colormap = flag;
%      title(plotname);
% end


% For 1< G < 10:

% for i=1:10
%      filename = ['Win_1G_' num2str(i) 'Q_1Winp_1Pexc_0#3.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      disp(training_output.acc);
%      subplot(2, 5, i);
%      imagesc(training_output.weights.static);
%      axis square;
%      title(plotname);
% end

% For G = 5:

% for i=1:9
%      filename = ['Win_1G_5Q_1Winp_1Pexc_0_' num2str(i) '.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      disp(training_output.acc);
%      subplot(3, 3, i);
%      imagesc(training_output.weights.static);
%      axis square;
%      title(plotname);
% end


% Calculating sum and total positive/negative static weights:

% G_vals = 1:0.5:5;
% for i=1:length(G_vals)
%      filename = ['Win_1G_' num2str(G_vals(i)) 'Q_1Winp_1Pexc_0.mat'];
%      load(filename);
%      fprintf(['Network ' num2str(i) ': \n'])
%      fprintf(['Test accuracy: ' num2str(training_output.acc) '\n'])
%      fprintf('sum: ')
%      fprintf('%f', sum(training_output.weights.static, 'all'));
%      fprintf('\n')
%      v = training_output.weights.static < 0;
%      w = training_output.weights.static > 0;
%      x = size(find(v));
%      y = size(find(w));
%      fprintf('Number of weights less than zero: ')
%      fprintf('%d', x(1));
%      fprintf('\n')
%      fprintf('Number of weights greater than zero: ')
%      fprintf('%d', y(1));
%      fprintf('\n\n')
% end


% Finds and calculates the number of connections for each neuron:

% G_vals = 1:0.5:5;
% for i=1:length(G_vals)
%      filename = ['Win_1G_' num2str(G_vals(i)) 'Q_1Winp_1Pexc_0.mat'];
%      load(filename);
%      connections = zeros(1, length(training_output.weights.static));
%      for j=1:length(training_output.weights.static)
%          row_connections = size(find(training_output.weights.static(j,:)));
%          connections(j) = row_connections(2);
%      end
%      plotname = ['Network ' num2str(i) 'Most connections: ' num2str(max(connections)) ', test accuracy: ' num2str(training_output.acc)];
%      subplot(3, 3, i);
%      histogram(connections);
%      title(plotname);
% end


% Finds and plots the eigenvalues of the static weight matrix:

%For G between 1 and 5:
 
G_vals = 1:0.5:5;
for i=1:length(G_vals)
     filename = ['Win_1G_' num2str(G_vals(i)) 'Q_1Winp_1Pexc_0.mat'];
     load(filename);
     plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
     eigenvalues = eig(training_output.weights.static);
     X = real(eigenvalues);
     Y = imag(eigenvalues);
     subplot(3, 3, i);
     scatter(X,Y, 4, '.', 'black');
     xlim([1.1*min(X) 1.1*max(X)])
     ylim([1.1*min(Y) 1.1*max(Y)])
     grid on;
     axis square;
     title(plotname);
end

%For G between 1 and 10:

% for i=1:10
%      filename = ['Win_1G_' num2str(i) 'Q_1Winp_1Pexc_0#3.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      eigenvalues = eig(training_output.weights.static);
%      X = real(eigenvalues);
%      Y = imag(eigenvalues);
%      subplot(2, 5, i);
%      scatter(X,Y, 4, '.', 'black');
%      xlim([1.1*min(X) 1.1*max(X)])
%      ylim([1.1*min(Y) 1.1*max(Y)])
%      grid on;
%      axis square;
%      title(plotname);
% end

% For G = 5:

% for i=1:9
%      filename = ['Win_1G_5Q_1Winp_1Pexc_0_' num2str(i) '.mat'];
%      load(filename);
%      plotname = ['Network ' num2str(i) ', test accuracy: ' num2str(training_output.acc)];
%      eigenvalues = eig(training_output.weights.static);
%      X = real(eigenvalues);
%      Y = imag(eigenvalues);
%      subplot(3, 3, i);
%      scatter(X,Y, 4, '.', 'black');
%      xlim([1.1*min(X) 1.1*max(X)])
%      ylim([1.1*min(Y) 1.1*max(Y)])
%      grid on;
%      axis square;
%      title(plotname);
% end









