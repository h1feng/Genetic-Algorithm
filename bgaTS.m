clc; clear; close all;

% ==== GA Parameters ====
popSize = 10;
nBits = 10;
nGenerations = 1000;
crossoverRate = 0.8;
mutationRate = 0.01;
xMin = -10;
xMax = 10;
nElite = 1;
tournamentSize = 2;

% ==== Fitness Function ====
fitnessFunc = @(x) -15 * sin(2 * x).^2 - (x - 2).^2 + 160;
decode = @(bin) xMin + bin2dec(num2str(bin)) * (xMax - xMin) / (2^nBits - 1);

% ==== Initial Population ====
pop = randi([0, 1], popSize, nBits);

% ==== For Plotting ====
bestFitnessHistory = zeros(1, nGenerations);
bestXHistory = zeros(1, nGenerations);

% ==== GA Main Loop ====
for gen = 1:nGenerations
    % Decode
    xVals = arrayfun(@(i) decode(pop(i, :)), 1:popSize);
    fitness = fitnessFunc(xVals);

    % Record best
    [bestFitness, idx] = max(fitness);
    bestFitnessHistory(gen) = bestFitness;
    bestXHistory(gen) = xVals(idx);

    % === Elitism ===
    [~, sortedIdx] = sort(fitness, 'descend');
    elites = pop(sortedIdx(1:nElite), :);

    % === Tournament Selection ===
    selected = zeros(size(pop));
    for i = 1:popSize
        candidates = randi(popSize, [1, tournamentSize]);
        [~, bestIdx] = max(fitness(candidates));
        selected(i, :) = pop(candidates(bestIdx), :);
    end

    % === Crossover ===
    for i = 1:2:popSize-1
        if rand < crossoverRate
            point = randi([1, nBits - 1]);
            p1 = selected(i, :);
            p2 = selected(i+1, :);
            selected(i, :) = [p1(1:point), p2(point+1:end)];
            selected(i+1, :) = [p2(1:point), p1(point+1:end)];
        end
    end

    % === Mutation ===
    for i = 1:popSize
        for j = 1:nBits
            if rand < mutationRate
                selected(i, j) = 1 - selected(i, j);
            end
        end
    end

    % === Insert Elites ===
    selected(1:nElite, :) = elites;

    % === Replace Population ===
    pop = selected;
end

% ==== Final Result ====
fprintf('Best x = %.5f\n', bestXHistory(end));
fprintf('Best fitness = %.5f\n', bestFitnessHistory(end));

% ==== Plot ====
figure('Color', 'w', 'Position', [100, 100, 600, 400]);

subplot(2,1,1);
xPlot = linspace(xMin, xMax, 1000);
yPlot = fitnessFunc(xPlot);
plot(xPlot, yPlot, 'b-', 'LineWidth', 2);
xlabel('X'); ylabel('Fitness value');
title('Fitness Function f(x)');
grid on;

subplot(2,1,2);
plot(1:nGenerations, bestFitnessHistory, 'm-', 'LineWidth', 2);
xlabel('Generation'); ylabel('Best fitness');
title('Best Fitness Over Generations (Tournament Selection)');
ylim([min(bestFitnessHistory)-2, max(bestFitnessHistory)+2]);
grid on;
