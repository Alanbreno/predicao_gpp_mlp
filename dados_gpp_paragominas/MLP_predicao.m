clear all;

loc1 = readmatrix('gpp_-2.92165_-47.38870.csv');
loc2 = readmatrix('gpp_-2.92304_-47.26719.csv');
loc3 = readmatrix("gpp_-2.92411_-47.32732.csv");
loc4 = readmatrix("gpp_-2.92499_-47.45306.csv");
loc5 = readmatrix('gpp_-2.96750_-47.45444.csv');
loc6 = readmatrix('gpp_-2.96992_-47.38774.csv');
loc7 = readmatrix('gpp_-2.97390_-47.26675.csv');
loc8 = readmatrix('gpp_-3.01545_-47.32965.csv');
loc9 = readmatrix('gpp_-3.01665_-47.26710.csv');
loc10 = readmatrix('gpp_-3.02046_-47.38980.csv');
loc11 = readmatrix('gpp_-3.02126_-47.45373.csv');
loc12 = readmatrix('gpp_-3.06282_-47.32748.csv');
loc13 = readmatrix('gpp_-3.06537_-47.26552.csv');
loc14 = readmatrix('gpp_-3.06592_-47.39200.csv');
loc15 = readmatrix('gpp_-3.06789_-47.45373.csv');

%13 anos de dados para treino
treino = [loc1(2:4749, 2); loc2(2:4749, 2); loc3(2:4749, 2); loc4(2:4749, 2); loc5(2:4749, 2);...
         loc6(2:4749, 2); loc7(2:4749, 2); loc8(2:4749, 2); loc9(2:4749, 2); loc10(2:4749, 2);...
         loc11(2:4749, 2); loc12(2:4749, 2); loc13(2:4749, 2); loc14(2:4749, 2); loc15(2:4749, 2)];

%5 anos de dados para validação
val = [loc1(4750:6575, 2); loc2(4750:6575, 2); loc3(4750:6575, 2); loc4(4750:6575, 2); loc5(4750:6575, 2);...
         loc6(4750:6575, 2); loc7(4750:6575, 2); loc8(4750:6575, 2); loc9(4750:6575, 2); loc10(4750:6575, 2);...
         loc11(4750:6575, 2); loc12(4750:6575, 2); loc13(4750:6575, 2); loc14(4750:6575, 2); loc15(4750:6575, 2)];

%4 anos para teste
teste = [loc1(6576:8036, 2); loc2(6576:8036, 2); loc3(6576:8036, 2); loc4(6576:8036, 2); loc5(6576:8036, 2);...
         loc6(6576:8036, 2); loc7(6576:8036, 2); loc8(6576:8036, 2); loc9(6576:8036, 2); loc10(6576:8036, 2);...
         loc11(6576:8036, 2); loc12(6576:8036, 2); loc13(6576:8036, 2); loc14(6576:8036, 2); loc15(6576:8036, 2)];


%treino
n = 6:71220;
P = [treino(n-1) treino(n-2) treino(n-3) treino(n-4) treino(n-5)]';
T = [treino(n)]';

%validação
n = 6:27390;
P1 = [val(n-1) val(n-2) val(n-3) val(n-4) val(n-5)]';
T1 = [val(n)]';

%teste
n = 6:21915;
P2 = [teste(n-1) teste(n-2) teste(n-3) teste(n-4) teste(n-5)]';
T2 = [teste(n)]';

a = minmax(P);

% Normalização de entrada
for i=1:71215
    P(1,i) = (P(1,i)-a(1,1))/(a(1,2) - a(1,1));
    P(2,i) = (P(2,i)-a(2,1))/(a(2,2) - a(2,1));
    P(3,i) = (P(3,i)-a(3,1))/(a(3,2) - a(3,1));
    P(4,i) = (P(4,i)-a(4,1))/(a(4,2) - a(4,1));
    P(5,i) = (P(5,i)-a(5,1))/(a(5,2) - a(5,1));
end

for i=1:27385
    P1(1,i) = (P1(1,i)-a(1,1))/(a(1,2) - a(1,1));
    P1(2,i) = (P1(2,i)-a(2,1))/(a(2,2) - a(2,1));
    P1(3,i) = (P1(3,i)-a(3,1))/(a(3,2) - a(3,1));
    P1(4,i) = (P1(4,i)-a(4,1))/(a(4,2) - a(4,1));
    P1(5,i) = (P1(5,i)-a(5,1))/(a(5,2) - a(5,1));
end

for i=1:21910
    P2(1,i) = (P2(1,i)-a(1,1))/(a(1,2) - a(1,1));
    P2(2,i) = (P2(2,i)-a(2,1))/(a(2,2) - a(2,1));
    P2(3,i) = (P2(3,i)-a(3,1))/(a(3,2) - a(3,1));
    P2(4,i) = (P2(4,i)-a(4,1))/(a(4,2) - a(4,1));
    P2(5,i) = (P2(5,i)-a(5,1))/(a(5,2) - a(5,1));
end

%
E = [P P1 P2];
S = [T T1 T2];
%
net = feedforwardnet(50);
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:71215;
net.divideParam.valInd = 71216:98600;
net.divideParam.testInd = 98601:120510;

[net, tr] = train(net, E, S);

a = sim(net, P2);

erroM = immse(a, T2);

