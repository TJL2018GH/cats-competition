%% data exploration for cats-competition

%% data importation
lbl=Subgroup;
X=Traincall'; % log2-fold changes with respect to a native ref. sample
X_sub=X(1:50,:);
lbl_sub=Subgroup(1:50);

%% data range
max(max(X_sub))
min(min(X_sub))

%% (partial) correlation network (metabolic network inference) is computationally unfeasible
cutoff=0.05;
[R1,ADJ]=PC(X_sub,cutoff);

%% correlations
[R(:,:),P(:,:)] = corrcoef(X_sub(:,:));

%% SOM of 3 classes
clc;
net = selforgmap([4 4]);
help selforgmap

%%
[net,tr] = train(net,X_sub');

