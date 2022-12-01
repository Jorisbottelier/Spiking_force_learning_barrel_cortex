% Creates plots for the initialized weight matrices.

for i=1:9
     filename = ['Win_1G_5Q_1Winp_1Pexc_0_' num2str(i) '.mat'];
     load(filename);
     plotname = ['Network test accuracy: ' num2str(training_output.acc) 'plotje nummero: ' num2str(i)];
     disp(training_output.acc);
     subplot(3, 3, i);
     imagesc(training_output.weights.static);
     axis square;
     title(plotname);
end