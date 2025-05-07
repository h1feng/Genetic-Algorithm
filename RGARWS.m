clc; clear; close all;

% ==== GA Parameters ====
popSize = 10;
nGenerations = 1000;
crossoverRate = 0.8;
mutationRate = 0.1;
xMin = -10;
xMax = 10;
nElite = 1;

% ==== Fitness Function ====
fitnessFunc = @(x) -15 * sin(2 * x).^2 - (x - 2).^2 + 160;

% ==== Initial Population ====
pop = xMin + (xMax - xMin) * rand(popSize, 1);  % real-valued initial population

% ==== For Plotting ====
bestFitnessHistory = zeros(1, nGenerations);
bestXHistory = zeros(1, nGenerations);

% ==== GA Main Loop ====
for gen = 1:nGenerations
    % Evaluate fitness
    fitness = fitnessFunc(pop);

    % Record best
    [bestFitness, idx] = max(fitness);
    bestFitnessHistory(gen) = bestFitness;
    bestXHistory(gen) = pop(idx);

    % === Elitism ===
    [~, sortedIdx] = sort(fitness, 'descend');
    elites = pop(sortedIdx(1:nElite));

    % === Selection: Roulette Wheel ===
    prob = fitness / sum(fitness);
    cumProb = cumsum(prob);
    newPop = zeros(size(pop));
    for i = 1:popSize
        r = rand;
        sel = find(cumProb >= r, 1, 'first');
        newPop(i) = pop(sel);
    end

    % === Crossover: Blend crossover (uniform style) ===
    for i = 1:2:popSize-1
        if rand < crossoverRate
            alpha = rand;  % blending factor
            p1 = newPop(i);
            p2 = newPop(i+1);
            newPop(i) = alpha * p1 + (1 - alpha) * p2;
            newPop(i+1) = alpha * p2 + (1 - alpha) * p1;
        end
    end

    % === Mutation: Gaussian noise ===
    for i = 1:popSize
        if rand < mutationRate
            mutationStrength = (xMax - xMin) * 0.1;
            newPop(i) = newPop(i) + mutationStrength * randn;
        end
        % Bound check
        newPop(i) = max(min(newPop(i), xMax), xMin);
    end

    % === Insert Elites ===
    newPop(1:nElite) = elites;

    % === Replace Population ===
    pop = newPop;
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
plot(1:nGenerations, bestFitnessHistory, 'r-', 'LineWidth', 2);
xlabel('Generation'); ylabel('Best fitness');
title('Best Fitness Over Generations (Real-coded GA)');
ylim([min(bestFitnessHistory)-2, max(bestFitnessHistory)+2]);
grid on;
